import os
import time
from collections import deque

import numpy as np
import torch

from core.arguments import get_args
from core.envs import make_vec_envs
from core.model import Policy, MultiHeadPolicy
from core.multi_action_heads import MultiActionHeads
from core.storage import RolloutStorage
from core import utils
from apievasion_rl.envs.utils.Detector import APIdetector

from algorithms.ppo import PPO


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.test_env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, no_obs_norm=args.no_obs_norm)


    action_history = np.zeros([312,312])
    if args.multi_action_head:
        head_infos = envs.get_attr("head_infos")[0]
        autoregressive_maps = envs.get_attr("autoregressive_maps")[0]
        action_type_masks = torch.tensor(envs.get_attr("action_type_masks")[0], dtype=torch.float32, device=device)
        action_heads = MultiActionHeads(head_infos, autoregressive_maps, action_type_masks,
                                        input_dim=args.hidden_size)
        # actor_critic = MultiHeadPolicy(envs.observation_space.shape, action_heads, use_action_masks=args.use_action_masks,
        #                     base_kwargs={'recurrent': args.recurrent_policy, 'recurrent_type': args.recurrent_type,
        #                                  'hidden_size': args.hidden_size})
        actor_critic = torch.load('./saved_models/ppo/7000_svm_noAPR.pt')[0]

        if args.multi_action_head:
            action_head_info = envs.get_attr("head_infos")[0]
        rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape,
                                  action_head_info=action_head_info, action_space=envs.action_space,
                                  recurrent_hidden_state_size=64,
                                  multi_action_head=args.multi_action_head)

        obs = envs.reset()

        rollouts.obs[0].copy_(obs)

        sample_count = 837
        evasions = 0
        total_pay = 0
        sample = 0
        while sample <= sample_count:
            for step in range(args.num_steps):
                with torch.no_grad():
                    if actor_critic.is_recurrent and actor_critic.base.recurrent_type == "LSTM":
                        recurrent_hidden_state_in = (rollouts.recurrent_hidden_states[step],
                                                  rollouts.recurrent_cell_states[step])
                    else:
                        recurrent_hidden_state_in = rollouts.recurrent_hidden_states[step]

                    if args.multi_action_head:
                        # update mask
                        if actor_critic.use_action_masks:
                            action_masks = envs.env_method(
                                "get_available_actions")  # build in zip so it returns [head_1(all_envs), head_2(all_envs), ...]
                            if args.multi_action_head:
                                action_masks = list(zip(*action_masks))
                                for i in range(len(rollouts.actions)):
                                    rollouts.action_masks[i][step].copy_(torch.tensor(action_masks[i]))
                        rollouts.to(device)

                        action_masks = [rollouts.action_masks[i][step] for i in range(len(rollouts.actions))]
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], recurrent_hidden_state_in, rollouts.masks[step],
                    action_masks=action_masks
                )

                action_history[action[0],action[1]] += 1

                obs, reward, done, infos = envs.step(action)
                payloads_rate = infos[0]['payload_rates']

                if done:
                    sample+=1

                    if  reward >= 10:
                        evasions += 1
                        total_pay += payloads_rate
                        break

    evasion_rate = (evasions / sample_count) * 100
    pay_rate = (total_pay / evasions)
    print(f"{evasion_rate}% samples evaded model with {pay_rate}.")
    print(evasions)
if __name__ == "__main__":
    main()