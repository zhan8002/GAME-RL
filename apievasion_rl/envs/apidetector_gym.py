import hashlib
import os
import random
import sys
from collections import OrderedDict
import gym
import numpy as np
from gym import spaces
from apievasion_rl.envs.controls import modifier
from apievasion_rl.envs.utils import Detector, interface
from apievasion_rl.envs.utils.Detector import APIdetector
from apievasion_rl.envs.utils.interface import cal_payloads_rate

import pandas as pd

module_path = '/home/omnisky/zhan/API/apievasion_rl/envs/controls'
APIHASH = np.load(os.path.join(module_path,"APIHash.npy"), allow_pickle=True).item()
APIINDEX = pd.read_csv(os.path.join(module_path,"api_312.csv"), header=None, index_col= 1, squeeze=True).to_dict()

# random.seed(0)
module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
model_type = 'rf' # choose target api detector 'rnn' or 'brnn' or 'lstm' or 'blstm' or 'gru' or 'bgru' or 'nn' or 'cnn'
model = APIdetector(model_type)
malicious_threshold = model.threshold
api_length = 200

class APIdetectorEnv(gym.Env):
    """Create MalConv gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sha256list,
        random_sample=True,
        maxturns=10,
        output_path="data/evaded/ember",
    ):
        super().__init__()
        self.available_sha256 = sha256list
        # self.action_space = spaces.Discrete(api_length)
        observation_high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-observation_high,
            high=observation_high,
            shape=(api_length, 8),
            dtype=np.int32,
        )
        self.maxturns = maxturns
        self.feature_extractor = model.extract
        self.output_path = output_path
        self.random_sample = random_sample
        self.history = OrderedDict()
        self.sample_iteration_index = 0

        self.output_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__),
                    ),
                ),
            ),
            output_path,
        )

        self.head_infos = [
            {"type": "categorical", "out_dim": 312},
            {"type": "categorical", "out_dim": 20}
        ]
        self.autoregressive_maps = [
            [-1],
            # [-1, 0]
            [-1]
        ]
        self.action_type_masks = [[1] for _ in range(312)]


    def step(self, action_ix):
        # Execute one time step within the environment
        self.turns += 1
        self._take_action(action_ix)
        self.observation_space = self.feature_extractor(self.sequence)
        self.score = model.predict_sample(self.observation_space)

        payloads_rate = cal_payloads_rate(self.l_sequence)

        if self.score < malicious_threshold:
            reward = 10.0
            episode_over = True
            self.history[self.sha256]["evaded"] = True
            self.history[self.sha256]["reward"] = reward


        elif self.turns >= self.maxturns:
            # game over - max turns hit
            # reward = self.original_score - self.score
            reward = 0
            episode_over = True
            self.history[self.sha256]["evaded"] = False
            self.history[self.sha256]["reward"] = reward
        else:
            # reward = self.original_score - self.score

            # query_reward = 10 * (1 - self.turns / self.maxturns)
            # payload_reward = 10 * (1 - payloads_rate)
            # reward = 0.05 * query_reward + 0.05*payload_reward

            reward = 0
            episode_over = False



        info = {
            "available_actions": self.get_available_actions(),
            "payload_rates": payloads_rate
        }

        if episode_over:
            # print(f"Episode over: reward = {reward}")
            evaded = self.history[self.sha256]["evaded"]
            print(f"Episode over: evaded = {evaded}")

        return self.observation_space, reward, episode_over, info

    def _take_action(self, action_ix):

        self.history[self.sha256]["actions"].append(action_ix)
        # pad_index = random.randint(1, 312)
        self.l_sequence = modifier.modify_sample(self.l_sequence, action_ix, self.turns)
        self.sequence = self.feature_extractor(self.l_sequence)



    def reset(self):
        # Reset the state of the environment to an initial state
        self.turns = 0
        while True:
            # grab a new sample (TODO)
            if self.random_sample:
                self.sha256 = random.choice(self.available_sha256)
            else:
                self.sha256 = self.available_sha256[
                    self.sample_iteration_index % len(self.available_sha256)
                ]
                self.sample_iteration_index += 1

            self.history[self.sha256] = {"actions": [], "evaded": False}
            self.sequence = interface.fetch_file(
                os.path.join(
                    module_path,
                    "utils/samples/",
                )
                + self.sha256,
            )

            self.l_sequence = np.insert(self.feature_extractor(self.sequence), 8, 100 * np.ones(200), axis=1)

            self.observation_space = self.feature_extractor(self.l_sequence)
            self.original_score = model.predict_sample(
                self.observation_space,
            )
            if self.original_score < malicious_threshold:
                # already labeled benign, skip
                continue

            break
        print(f"Sample: {self.sha256}")
        return self.observation_space

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        pass


    def get_available_actions(self):
        avail_acs_all = []
        for i in range(len(self.head_infos)):
            ac_dim = self.head_infos[i]["out_dim"]

            if i ==0:
                avail_acs_all.append(np.zeros((ac_dim,)))
            else:
                avail_acs_all.append(np.ones((ac_dim,)))

        avail_loc = np.where(self.l_sequence[0:api_length -1, 8] == 100)[0]
        for loc in avail_loc:
            avail_api = self.l_sequence[loc, :8]
            for id in range(1, len(APIHASH) + 1):
                if (APIHASH[APIINDEX[id]] == avail_api).all():
                    avail_acs_all[0][id-1] = 1

        return avail_acs_all
