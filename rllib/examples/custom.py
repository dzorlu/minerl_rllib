
import argparse

from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Space
from gym.spaces import Tuple as TupleSpace

from ray.rllib.utils.framework import try_import_tf, try_import_torch
torch, nn = try_import_torch()

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC

from ray import tune
import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID, Dict, Tuple
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import restore_original_dimensions, ModelV2


#https://docs.ray.io/en/latest/rllib-training.html#curriculum-learning
# environment class has `set_phase()` method that adjusts the task difficulty over time
# multiagent env
# (1) as a user, you define the number of policies available up front, 
# (2) a function that maps agent ids to policy ids.

"""A wrapper that puts the previous action, reward, and
  observation vector into observation
"""
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import NamedTuple

import tqdm
from collections import OrderedDict

import minerl
import numpy as np
import pickle
import os
from collections import Counter

import gym

import coloredlogs, logging
coloredlogs.install(logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=None)
parser.add_argument("--stop-timesteps", type=int, default=5000000)
parser.add_argument("--stop-reward", type=float, default=0.1)

NB_SUBTASKS = 5
NUM_ACTIONS = 30
ENV_NAME = "MineRLObtainDiamondVectorObf-v0"
MINERL_DATA_ROOT = '/data'
# speeds up the env
os.environ["OMP_NUM_THREADS"] = "1"

def create_clusters(num_actions=NUM_ACTIONS, num_trajectories=5):
    k_means = KMeans(n_clusters=num_actions, random_state=0)
    print('create kmeans mapping')
    model_path='/home/deniz/ray_results/kmeans/'
    file_path = os.path.join(model_path, 'k_means.pkl')

    # replay trajectories
    data = minerl.data.make(ENV_NAME, 
                            data_dir=MINERL_DATA_ROOT, 
                            num_workers=1,
                            worker_batch_size=4)
    trajectories = data.get_trajectory_names()[:num_trajectories]
    actions = list()
    for t, trajectory in enumerate(trajectories):
        logger.info({str(t): trajectory})
        for i, (state, a, r, _, done, meta) in enumerate(data.load_data(trajectory, include_metadata=True)):    
            action = a['vector'].reshape(1, 64)
            actions.append(action)
    actions = np.vstack(actions)
    k_means.fit(actions)
    logger.info({'finished': len(actions)})
    del actions
    pickle.dump(k_means, open(file_path, 'wb'))
    #logger.info({'persisted k-means under': file_path})
    print('persisted k-means')
    return k_means



class MineRLEnv(gym.Wrapper, MultiAgentEnv):
    """
    at each hierarchy, the reward the subpolicy is seeking
    and number of times the reward needs to be obtained
    to move on to the next subpolicy.

    phase determines the difficulty in curriculum learning.
    at each phase, a new critic is initialized.
    in each phase, there is a set of subtasks that are active.

    """
    def __init__(self, env, config):
        print("env")
        print(env)
        print("config")
        print(config)
        gym.Wrapper.__init__(self, env)
        # curriculum learning ix
        # couple w/ env_rewards, ix determines the `task`
        self.phase = 0
        self.env_times_reward = Counter()

        num_actions = config.get('num_actions')
        num_subtasks = config.get('num_subtasks')

        env_rewards = {i: {'reward': 2**i, 'times': 1} for i in range(num_subtasks)}
        env_rewards[0]['times'] = 64 # #1 needs 64 logs to complete.
        self.env_expected_rewards = env_rewards

        # sub-policy ix. 
        # denotes which subpolicy is active.
        self.subpolicy_ix = 0
        self.increment_subpolicy = False


        # observation and action space
        # do not need the extra `stop` action.
        self.observation_space = self.env.observation_space
        # action space is mapped to a discrete function.
        self.action_space = gym.spaces.Discrete(num_actions)

        # build demonstation dataset at initialization
        self.build_replay_data()
        # discrete control
        self.k_means = config.get('kmeans')
        print("k-means completed.")
    
    def build_replay_data(self):
        pass
    
    def increment_phase(self):
        """
        increment the phase when a certain score is reached.
        called by `on_train_result`
        this should prompt discard / rebuilding of the replay data.
        """
        #https://docs.ray.io/en/latest/rllib-training.html#curriculum-learning
        self.phase += 1
        self.build_replay_data()

    def maybe_increment_subpolicy(self, done):
        if self.increment_subpolicy:
            self.subpolicy_ix += 1
            self.increment_subpolicy = False

    def reshape_reward_and_done(self, r, done):
        """
        called by `step` function.
        increments the subpolicy ix where the next subpolicy model takes over.
        reshapes reward and one
        
        phase subpolicy times 
        -----   ------   ----
        0           0       1  -> reward, not done
        0           0       64 -> reward, done
        1           1       1  -> reward, done
        1           0       1  -> no reward, not done. reward at the end.

        """
        if self.env_expected_rewards[self.subpolicy_ix]['reward'] == r:
            self.env_times_reward[self.subpolicy_ix] += 1
        if self.env_expected_rewards[self.subpolicy_ix]['times'] ==  self.env_times_reward[self.subpolicy_ix]:
            self.increment_subpolicy = True #activate the next subpolicy (if not done)
            done[self.subpolicy_ix] = True

        reward = 0
        # curriculum learning. set done to TRUE if last phase is solved.
        if self.env_expected_rewards[self.phase]['reward'] == r:
            print('reward!!!!')
            reward = 1 #normalize to 1.
            if self.env_expected_rewards[self.phase]['times'] == self.env_times_reward[self.phase]:
                done["__all__"] = True
        return reward, done

    def reset(self):
        self.subpolicy_ix = 0
        self.increment_subpolicy = False
        obs = {}
        obs[self.subpolicy_ix] = self.env.reset()
        return obs

    def convert_action(self, action: np.array) -> OrderedDict:
        # map to cont action space that env demands
        return OrderedDict({'vector': self.k_means.cluster_centers_[action]})
        #return self.env.action_space.sample()

    def step(
            self, action_dict: Dict[AgentID, int]
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # Convert action_dict (used by RLlib) to a list of actions (used by
        # minerl environment)
        obs = {}
        rewards = {}
        done = {"__all__": False}
        info = {}
        logger.info('stepping in the environment..')
        for agent_id, action in action_dict.items():
            # allow 'active' agent to take a step.
            if agent_id == self.subpolicy_ix:
                cont_action = self.convert_action(action)
                _obs, r, _done, _ = self.env.step(cont_action)
                if _done:
                    done["__all__"] = True
                reward, done = self.reshape_reward_and_done(r, done)
                # map to dict
                obs[agent_id] = _obs
                rewards[agent_id] = reward
                # incrementing subpolicy and marking the current subpolicy as done.
                self.maybe_increment_subpolicy(reward)
        
        return obs, rewards, done, {}


###########
# model(s)#
###########

# Shared across policies
class VisionNet(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 filters,
                 num_outputs,
                 input_shape):
        super(VisionNet, self).__init__()
        layers = []
        (w, h, in_channels) = input_shape
        in_size = [w, h]
        layers = []
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn="relu"))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        # final FC layer.
        layers.append(
            SlimConv2d(
                32,
                num_outputs,
                [1,1],
                stride,
                None,  # padding=valid
                activation_fn='relu'))
        # squeeze into [B, num_outputs]
        layers.append(nn.Flatten(start_dim=1, end_dim=3))
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)



VISUAL_SIZE_OUT = 100
VISUAL_SIZE_IN = [64,64,3]
conv_filters =  [
    [32, [3, 3], 2],
    [32, [3, 3], 2],
    [32, [3, 3], 2],
    [32, [3, 3], 2],
    [32, [3, 3], 2],
    [32, [3, 3], 2],
    [32, [2, 2], 1],
]

VISION_MODEL = VisionNet(filters=conv_filters, num_outputs=VISUAL_SIZE_OUT, input_shape=VISUAL_SIZE_IN)

# TODO: MAKE THIS DEEPER
VECTOR_SIZE_IN = 64
VALUE_BRANCH_IN = VISUAL_SIZE_OUT + VECTOR_SIZE_IN #SHAPE OF THE VECTOR VARIABLE
VALUE_BRANCH = SlimFC(in_size=VALUE_BRANCH_IN, out_size=1, activation_fn=None, initializer=torch.nn.init.xavier_uniform_)

class TaskModel(TorchRNN, nn.Module):
    """
    A recurrent model that shares the pov and critic modules.
    """

    def __init__(self, 
                 obs_space, 
                 action_space,
                 num_outputs, 
                 model_config,
                 name,
                 visual_size_in,
                 observation_size_in,
                 input_size,
                 lstm_state_size=64):
        TorchRNN.__init__(self, 
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )
        nn.Module.__init__(self)

        self.lstm_state_size = lstm_state_size
        self.cnn_shape = visual_size_in
        self.obs_shape = observation_size_in
        self.input_size = input_size

        # share the vision model and the value branch.
        self.vision_encoder = VISION_MODEL
        self.value_branch = VALUE_BRANCH

        self.lstm = nn.LSTM(
            self.input_size, self.lstm_state_size, batch_first=True)

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits = SlimFC(self.lstm_state_size, self.num_outputs)
        
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        inputs: TensorType
        overriding the `forward` method to accomodate nested observation
        """
        inputs = restore_original_dimensions(
            inputs, self.obs_space, self.framework
        )
        # TODO: Normalize?
        pov = inputs['pov']
        vision_in = torch.reshape(pov, [-1] + self.cnn_shape)
        vision_in = vision_in.permute(0, 3, 1, 2)
        vision_out = self.vision_encoder(vision_in) #[B x T, visual_size_out]
        # Concat
        vec = inputs['vector']
        vec = torch.reshape(vec, [-1] + [self.obs_shape]) ##[B x T, obs_shape]
        vec = torch.cat((vision_out, vec), 1) # [B, visual_size_out + obs_shape]
        vec = torch.reshape(
            vec, 
            [pov.shape[0], pov.shape[1], vec.shape[-1]])
        self._features = vec #branches out here # [B, T, visual_size_out + obs_shape]
        #print(self._features.shape)
        # Flatten
        if len(state[0].shape) == 2:
            # TODO: IS THIS RIGHT?
            # print("**")
            # print(state[0].shape)
            state[0] = state[0].unsqueeze(0)
            # print(state[0].shape)
            # print("**")
            state[1] = state[1].unsqueeze(0)
        # Forward through LSTM.
        _features, [h, c] = self.lstm(self._features, state)
        # Forward LSTM out through logits layer and value layer.
        logits = self.logits(_features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        """
        A separate branch to evaluate the value at a given space.
        """
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            np.zeros(self.lstm_state_size),
        ] * 2 # h,c
        return h

# auxillary functions
# def on_train_result(info):
#     result = info["result"]
#     if result["episode_reward_mean"] > 200:
#         phase = 2
#     elif result["episode_reward_mean"] > 100:
#         phase = 1
#     else:
#         phase = 0
#     trainer = info["trainer"]
#     trainer.workers.foreach_worker(
#         lambda ev: ev.foreach_env(None)
#     )

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # register environment
    
    def _wrap():
        import minerl
        return gym.make(ENV_NAME)
    register_env("minecraft", 
        lambda config: MineRLEnv(env=_wrap(),config=config))
    # register model
    logger.info("registering the model")
    ModelCatalog.register_custom_model(
        "task_model", TaskModel)

    logger.info("define action and obs space")
    # pov:Box(low=0, high=255, shape=(64, 64, 3)), 
    # vector:Box(low=-1.2000000476837158, high=1.2000000476837158, shape=(64,))
    obs_space = gym.spaces.Dict({
        "pov": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3)),
        "vector": gym.spaces.Box(low=-1.2, high=1.2, shape=(64, ))
    })

    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    
    # None -> use the same policy (VTrace Policy)
    policies = {f"policy_{i}": (None, obs_space, act_space, {}) for i in range(NB_SUBTASKS)}

    # https://docs.ray.io/en/master/rllib-training.html#scaling-guide
    config = {
        "env": 'minecraft',
        "env_config": {
            "num_actions": NUM_ACTIONS, #number of discrete actions
            "kmeans": create_clusters(),
            "num_subtasks": NB_SUBTASKS,
        },
        # "callbacks": {
        #     "on_train_result": on_train_result,
        # },
        "num_gpus": 1,
        "model": {
            "custom_model": "task_model",
            "custom_model_config": {
                "input_size": VALUE_BRANCH_IN, # input / outsize of the LSTM
                "visual_size_in": VISUAL_SIZE_IN, #shape of POV observation
                "observation_size_in": VECTOR_SIZE_IN, #shape of observation vector
            },
        },
        "multiagent": {
            "policies": policies, # separete sequential sub-policies
            "policy_mapping_fn": (lambda x: f"policy_{x}"),
        },
        "num_workers": 1,  # parallelism. number of rollout actors.
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 5,
        "framework": "torch",
        "train_batch_size": 600,
        "rollout_fragment_length": 50,
    }

    stop = {
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    logger.info("start tuning..")
    results = tune.run("IMPALA", 
                       checkpoint_freq=20, #number of iterations. 1 iter = .train() call
                       keep_checkpoints_num=10,
                       config=config, 
                       stop=stop,
                       verbose=1)

