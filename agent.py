import numpy as np
import torch
from torch.distributions import Categorical
from qmix import QMIX

# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.policy = QMIX(args)
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None):
        inputs = obs.copy()
        # avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value

        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value

        q_value[avail_actions == -1] = - float("inf")
        print("epsilon",epsilon)
        if np.random.uniform() < epsilon:
            action = np.random.choice([0,1,2])  # action是一个整数
        else:
            action = torch.argmax(q_value)

        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch1, batch2, train_step, episode_number, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len1 = self._get_max_episode_len(batch1)
        max_episode_len2 = self._get_max_episode_len(batch2)
        max_episode_len = max(max_episode_len1,max_episode_len2)
        for key in batch1.keys():
            if key != 'z':
                batch1[key] = batch1[key][:, :max_episode_len]
        for key in batch2.keys():
            if key != 'z':
                batch2[key] = batch2[key][:, :max_episode_len]
        self.policy.learn(batch1, batch2, max_episode_len, train_step, epsilon)
        if episode_number > 0 and episode_number % self.args.save_cycle == 0:
            self.policy.save_model(episode_number)


