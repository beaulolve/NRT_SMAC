from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import csv
import os


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        env_name = args.env_args['env_name']
        if env_name == 'SC2':
            env_name += '_'
            env_name += args.env_args['map_name']

        self.csv_dir = f'./csv_files/{env_name}/fresh_{args.scaler_fresh}_dim_{args.state_vae_latent_dim}_check_{args.check_point_interval}_add_{args.episode_add_interval}_vae_buffer_{args.state_vae_train_buffer}_forget_prop_{args.forget_prop}_alpha_{args.alpha_min}_{args.alpha_max}_logp_w_{args.forget_logp_punish_weight}_sample_{args.calculate_reward_sample}_{args.already_forget_sample}_CDS_{args.beta}_{args.beta1}_{args.beta2}_localq_norm_w_{args.localq_norm_w}'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def writereward(self, win_rate, step):
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, win_rate])
        else:
            with open(self.csv_path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'win_rate'])
                csv_write.writerow([step, win_rate])

    def run(self, test_mode=False, writer=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        state_list, action_list = [], []

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            state_list.append(self.env.get_state())

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            action_list.append(actions)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        state_list.append(self.env.get_state())
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                         for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            env_name = self.args.env_args['env_name']
            if env_name == 'SC2':
                cur_returns_mean = cur_stats['battle_won'] / \
                    cur_stats['n_episodes']
                print(cur_stats)
                print('=' * 30)
            else:
                cur_returns_mean = np.array(
                    [0 if item <= 0 else 1 for item in cur_returns]).mean()

            if writer:
                writer.add_scalar(f"eval_mean", cur_returns_mean, self.t_env)
            self.writereward(cur_returns_mean, self.t_env)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, torch.tensor(state_list[:-1]), torch.tensor(state_list[1:]), torch.cat(action_list, dim=0)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v/stats["n_episodes"], self.t_env)
        stats.clear()
