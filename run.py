import copy as cp
import datetime
import os
import pprint
import random
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from components.episode_buffer import ReplayBuffer, ReplayBuffer_Prior
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from modules.VAE.vae import BetaVAE
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = args.GPU if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}/{}__{}".format(args.env_args['map_name'], "NRT",
                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs")
        env_name = args.env_args['env_name']

        tensorboard_dir = f'{tb_logs_direc}/{env_name}/seed_{args.seed}'
        logger.setup_tb(tensorboard_dir)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]
    print(args.state_shape - args.n_agents * args.n_actions)
    print('=' * 30)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if 'prior' in args.name:
        buffer_prior = ReplayBuffer_Prior(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                          args.burn_in_period,
                                          preprocess=preprocess,
                                          device="cpu" if args.buffer_cpu_only else args.device,
                                          alpha=args.alpha)

        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    else:
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # here add a list for storing
    all_state_list = []
    forget_state_list = []
    forget_data_list = []
    forget_vae_list = []
    forget_vae_device_list = []
    episode_num = 0
    episode_add_interval = args.episode_add_interval

    # Learner
    learner = le_REGISTRY[args.learner](
        mac, buffer.scheme, logger, args, all_state_list, forget_state_list, forget_vae_device_list)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    on_policy_episode = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))

    state_vae_store_cur = []
    check_point_list = []
    last_add_check_point = 0
    cur_class = 1

    env_name = args.env_args['env_name']
    if env_name == 'SC2':
        env_name += '_'
        env_name += args.env_args['map_name']

    writer = SummaryWriter(
        logdir=f"runs/{env_name}/fresh_{args.scaler_fresh}_dim_{args.state_vae_latent_dim}_check_{args.check_point_interval}_add_{args.episode_add_interval}_vae_buffer_{args.state_vae_train_buffer}_forget_prop_{args.forget_prop}_alpha_{args.alpha_min}_{args.alpha_max}_logp_w_{args.forget_logp_punish_weight}_sample_{args.calculate_reward_sample}_{args.already_forget_sample}_CDS_{args.beta}_{args.beta1}_{args.beta2}_localq_norm_w_{args.localq_norm_w}/seed_{args.seed}")

    cur_state_vae = BetaVAE(args.device, args.state_shape - args.n_agents * args.n_actions,
                            args.state_vae_latent_dim)
    cur_state_vae_optimiser = th.optim.Adam(cur_state_vae.parameters())
    state_vae_on_policy_train = None

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch, state, _, _ = runner.run(test_mode=False)
        state = state[:, :-args.n_agents * args.n_actions]

        episode_num += 1
        if state_vae_on_policy_train is not None:
            state_vae_on_policy_train = th.cat(
                [state_vae_on_policy_train, state], dim=0)
        else:
            state_vae_on_policy_train = state

        if state_vae_on_policy_train.shape[0] > args.state_vae_train_buffer:
            state_vae_on_policy_train = state_vae_on_policy_train.to(
                args.device).float()

            for _ in range(args.state_vae_train_epoch):
                sampler = BatchSampler(
                    SubsetRandomSampler(
                        range(state_vae_on_policy_train.shape[0])),
                    args.state_vae_train_batch,
                    drop_last=True)

                for indices in sampler:
                    loss = cur_state_vae.get_loss(
                        state_vae_on_policy_train[indices])

                    cur_state_vae_optimiser.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        cur_state_vae.parameters(), args.grad_norm_clip)
                    cur_state_vae_optimiser.step()

            learner.refresh_cur_logp_state(cur_state_vae)
            state_vae_on_policy_train = None

        # collect on-policy data for calculating expectation (middle buffer: 100000)
        state_vae_store_cur.append(state)
        if len(state_vae_store_cur) > 1:
            while(th.cat(state_vae_store_cur, dim=0).shape[0] > args.check_point_interval):
                state_vae_store_cur.pop(0)

        # collect all data for forbidding forgetting (infinite)
        if episode_num % episode_add_interval == 0:
            all_state_list.append(state)

        # analyze whether forgetting
        if runner.t_env - last_add_check_point > args.check_point_interval:
            cur_data = th.cat(state_vae_store_cur, dim=0).float()
            cur_data_device = cur_data.to(args.device)

            state_vae = BetaVAE(args.device, args.state_shape - args.n_agents * args.n_actions,
                                args.state_vae_latent_dim)
            state_vae_optimiser = th.optim.Adam(
                state_vae.parameters())

            # 23/04/25 change from 16 to 4
            for _ in range(4):
                sampler = BatchSampler(SubsetRandomSampler(
                    range(cur_data_device.shape[0])), 512, drop_last=True)

                for indices in sampler:
                    loss = state_vae.get_loss(cur_data_device[indices])
                    state_vae_optimiser.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        state_vae.parameters(), args.grad_norm_clip)
                    state_vae_optimiser.step()

            state_vae_cpu = BetaVAE('cpu', args.state_shape - args.n_agents * args.n_actions,
                                    args.state_vae_latent_dim)
            state_vae_cpu.load_state_dict(state_vae.state_dict())

            if check_point_list != []:
                for t, item in enumerate(check_point_list):
                    with th.no_grad():
                        checkpoint_time = (t + 1) * args.check_point_interval

                        # use sample decoder loss to represent logp
                        KL_old_cur_sample_decoder = item['vae'].get_logp(
                            item['data']) - state_vae_cpu.get_logp(item['data'])
                        KL_cur_old_sample_decoder = state_vae_cpu.get_logp(
                            cur_data) - item['vae'].get_logp(cur_data)

                        KL_old_cur_sample_decoder = KL_old_cur_sample_decoder.mean().to('cpu')
                        KL_cur_old_sample_decoder = KL_cur_old_sample_decoder.mean().to('cpu')

                        JS_sample_decoder = KL_old_cur_sample_decoder + KL_cur_old_sample_decoder

                    writer.add_scalar(
                        f"checkpoint_time_{checkpoint_time}/JS_sample_decoder", JS_sample_decoder, runner.t_env)

                    if check_point_list[t]['max'] / check_point_list[t]['min'] > args.forget_prop \
                            and check_point_list[t]['max'] / JS_sample_decoder > args.forget_prop \
                    and check_point_list[t]['t_B'] != None \
                    and check_point_list[t]['can_add'] \
                        and check_point_list[t]['max'] > args.forget_min_max \
                            and check_point_list[t]['min'] < args.forget_max_min:

                        can_add = True
                        if forget_vae_list == []:
                            pass
                        else:
                            with th.no_grad():
                                for forget_id in range(len(forget_vae_list)):
                                    KL_old_already = item['vae'].get_logp(
                                        item['data']) - forget_vae_list[forget_id].get_logp(item['data'])
                                    KL_already_old = forget_vae_list[forget_id].get_logp(
                                        forget_data_list[forget_id]) - item['vae'].get_logp(forget_data_list[forget_id])

                                    KL_old_already = KL_old_already.mean()
                                    KL_already_old = KL_already_old.mean()
                                    JS_already = KL_old_already + KL_already_old

                                    if JS_already < args.forget_old_already:
                                        can_add = False
                                        check_point_list[t]['can_add'] = False
                                        break

                        if can_add:
                            cur_class += 1
                            forget_vae_list.append(
                                check_point_list[t]['vae'])
                            forget_vae_device_list.append(
                                cp.deepcopy(check_point_list[t]['vae']).to(args.device))
                            forget_state_list.append(
                                check_point_list[t]['list'])
                            forget_data_list.append(
                                check_point_list[t]['data'])

                    if check_point_list[t]['min'] > JS_sample_decoder:
                        check_point_list[t]['min'] = max(
                            JS_sample_decoder, 0.001)

                    if check_point_list[t]['max'] < JS_sample_decoder:
                        check_point_list[t]['max'] = max(
                            JS_sample_decoder, 0.001)
                        check_point_list[t]['t_B'] = cp.deepcopy(
                            state_vae).to('cpu')

            writer.add_scalar(f"revisit", cur_class - 1, runner.t_env)
            del cur_data_device

            # add cur check point
            check_point_list.append(
                {'vae': cp.deepcopy(state_vae).to('cpu'),
                 'data': cp.deepcopy(cur_data),
                 'list': cp.deepcopy(state_vae_store_cur),
                 't_B': None,
                 'max': 0.001,
                 'min': 100,
                 'can_add': True,
                 })
            last_add_check_point = runner.t_env

        if 'prior' in args.name:
            buffer.insert_episode_batch(episode_batch)
            buffer_prior.insert_episode_batch(episode_batch)
        else:
            buffer.insert_episode_batch(episode_batch)

        for _ in range(args.num_circle):

            if buffer.can_sample(args.batch_size):

                if 'prior' in args.name:
                    idx, episode_sample = buffer_prior.sample(args.batch_size)
                else:
                    episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if 'prior' in args.name:
                    update_prior = learner.train(
                        episode_sample, runner.t_env, episode)
                    buffer_prior.update_priority(
                        idx, update_prior.to('cpu').tolist())
                else:
                    learner.train(episode_sample, runner.t_env, episode)

            # on policy update prediction networks
            if "CDS" in args.learner:
                if episode - on_policy_episode >= args.on_policy_batch:
                    if buffer.on_policy_can_sample(args.on_policy_batch) \
                            and buffer.on_policy_can_sample(args.batch_size):

                        if args.ifon_sample:
                            episode_sample = buffer.on_policy_sample(
                                args.on_policy_batch)
                        else:
                            episode_sample = buffer.sample(args.batch_size)

                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)

                        learner.train_predict(episode_sample, runner.t_env)
                        on_policy_episode = episode

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True, writer=writer)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            if args.double_q:
                os.makedirs(save_path + '_x', exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run * args.num_circle

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.infoing(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
