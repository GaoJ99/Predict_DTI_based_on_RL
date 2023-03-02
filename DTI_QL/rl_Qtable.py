import numpy as np
import pandas as pd
from environment_model import EnvModel

class QLearningTable:
    def __init__(self, actions, action_spaces, learning_rate, reward_decay, e_greedy):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.action_spaces = action_spaces
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 生成q_table

    def choose_action(self, observation, count):
        self.check_state_exist(str(observation))  # 检测当前状态是否存在于Q表中，没有的话就在Q表中增加这个节点
        # 动作选择
        # while True:
        if np.random.uniform() > self.epsilon**count:  # epsilon等于0.9, 90%概率Q函数决定行动
            # 选择最佳的动作
            # print("self.epsilon**count:", self.epsilon**count)
            state_action = self.q_table.loc[str(observation), :]  # 取出当前observation所在的所有动作

            # 当最大Q值对应多个动作时，随机选择动作
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)  # 返回的是一个数字
        else:
            # 10%概率随机选择一个动作，探索其他可能
            # while True:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, done):
        # self.check_state_exist(s_)  # 检查是否存在下一个状态
        q_predict = self.q_table.loc[s, a]  # 获得当前状态，当前动作对应的预测分数
        if not done:  # done为True表示下个状态不是终点
            self.check_state_exist(s_)  # 检查是否存在下一个状态
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - q_predict)  # update


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

