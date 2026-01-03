import numpy as np
import os
from rollout import RolloutWorker
from agent import Agents
from replay_buffer1 import ReplayBuffer1
from replay_buffer1 import ReplayBuffer2
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer1 = ReplayBuffer1(args)
        self.buffer2 = ReplayBuffer2(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.step_numbers = []
        self.mean_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        episode_rewards = []
        max_rewardsingle = -10
        episode_number = 0
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(1, time_steps))
            print('Run {}'.format(time_steps // self.args.evaluate_cycle))
            if episode_number // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward, step_number, mean_reward,_ = self.evaluate()
                print('win_rate is ', win_rate)
                self.step_numbers.append(step_number)
                self.win_rates.append(win_rate)
                self.mean_rewards.append(mean_reward)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episode_number = episode_number + 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward1, win, steps, mean_reward1,_ = self.rolloutWorker.generate_episode(episode_number)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            if (episode_reward1 > max_rewardsingle * 0.3):
                self.buffer1.store_episode(episode_batch)
            else:
                self.buffer2.store_episode(episode_batch)
            if self.buffer1.current_size > 0 and self.buffer2.current_size > 0:
                for train_step in range(self.args.train_steps):
                    mini_batch1 = self.buffer1.sample(min(self.buffer1.current_size, 16))
                    mini_batch2 = self.buffer2.sample(min(self.buffer2.current_size, 16))
                    self.agents.train(mini_batch1, mini_batch2, train_steps, episode_number)
                    train_steps += 1
            if episode_reward1 > max_rewardsingle:
                max_rewardsingle = episode_reward1
            print("self.win_rates", self.win_rates)
            print("self.episode_rewards", self.episode_rewards)
            print("self.mean_rewards", self.mean_rewards)
            print("self.step_numbers", self.step_numbers)

        win_rate, episode_reward, step_number, mean_reward,_ = self.evaluate()
        self.step_numbers.append(step_number)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        total_speed3 = 0
        win_number = 0
        step_numbers = 0
        episode_rewards = 0
        mean_rewards = 0
        totalepisode_rewards = []
        totalstep_numbers = []
        totalepisode_speeds = []
        for epoch in range(self.args.evaluate_epoch):
            # epoch = epoch + 30
            print("epoch", epoch)
            _, episode_reward, win_tag, step, mean_reward,total_speed = self.rolloutWorker.generate_episode(epoch, evaluate=True) ###33
            episode_rewards += episode_reward
            mean_rewards += mean_reward
            if win_tag:
                step_numbers += step
            if win_tag:
                win_number += 1
            if win_tag:
                total_speed3 += total_speed
            totalepisode_rewards.append(episode_reward)
            if win_tag:
                totalstep_numbers.append(step)
            if win_tag:
                totalepisode_speeds.append(total_speed)
            print("totalepisode_rewards", totalepisode_rewards)
            print("totalstep_numbers", totalstep_numbers)
            print("totalepisode_speeds", totalepisode_speeds)
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, step_numbers / win_number, mean_rewards / self.args.evaluate_epoch, total_speed3/win_number

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 2, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')


        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        # np.save(self.save_path + '/step_numbers_{}'.format(num), self.step_numbers)
        plt.close()









