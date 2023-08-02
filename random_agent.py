import os
import gym
import numpy as np
import argparse
import pickle
import random
import apievasion_rl
from apievasion_rl.envs.utils.interface import cal_payloads_rate


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1100)
parser.add_argument('--eps_shapley', type=bool, default=False) # when epsilon explore, add shapley value prior instead random (ES)
parser.add_argument('--per', type=bool, default=True) #  use prioried experience replay instead random
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/SOREL_PSP/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

args = parser.parse_args()
import pandas as pd
sequence_length = 200
module_path = '/home/omnisky/zhan/API/apievasion_rl/envs/controls'
APIHASH = np.load(os.path.join(module_path,"APIHash.npy"), allow_pickle=True).item()
APIINDEX = pd.read_csv(os.path.join(module_path,"api_312.csv"), header=None, index_col= 1, squeeze=True).to_dict()

def get_available_actions(l_sequence):
    avail_acs_all = []
    for i in range(2):
        ac_dim = 312

        if i == 0:
            avail_acs_all.append(np.zeros((ac_dim,)))
        else:
            avail_acs_all.append(np.ones((ac_dim,)))

    avail_loc = np.where(l_sequence[0:sequence_length - 1, 8] == 100)[0]
    for loc in avail_loc:
        avail_api = l_sequence[loc, :8]
        for id in range(1, len(APIHASH) + 1):
            if (APIHASH[APIINDEX[id]] == avail_api).all():
                avail_acs_all[0][id - 1] = 1

    ## test##
    # avail_acs_all[0][0] = 1
    return avail_acs_all

class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self):
        self.aval_action = None

    def choose_action(self, label_sequence):
        avail = get_available_actions(label_sequence)
        hook = random.choice(np.where(avail[0]==1)[0])
        pad = random.choice(np.where(avail[1]==1)[0])
        action = [hook, pad]
        return action

def main():


    print('---------- start test ------------')

    env_test = gym.make('APIdetector-test-v0')

    agent = RandomAgent()

    episode_count = 837


    max_turn = env_test.maxturns

    evasions = 0
    evasion_history = {}
    total_pay = 0

    for i in range(episode_count):
        total_reward = 0
        done = False
        env_test.reset()
        sha256 = env_test.sha256
        num_turn = 0


        while num_turn < max_turn:

            action = agent.choose_action(env_test.l_sequence)# random agent
            obs, reward, done, infos = env_test.step(action)
            total_reward += reward

            num_turn = env_test.turns

            payloads_rate = infos['payload_rates']

            if done and reward >= 10.0:
                evasions += 1
                total_pay += payloads_rate
                break

            elif done:
                break


    # Output metrics/evaluation stuff
    evasion_rate = (evasions / episode_count) * 100
    pay_rate = (total_pay / evasions)
    print(f"{evasion_rate}% samples evaded model with {pay_rate}.")
    #
    # # write evasion_history to txt file
    # file = open('history_sorel2.txt', 'w')
    # for k, v in evasion_history.items():
    #     file.write(str(k) + ' ' + str(v) + '\n')
    # file.close()



if __name__ == '__main__':
    main()