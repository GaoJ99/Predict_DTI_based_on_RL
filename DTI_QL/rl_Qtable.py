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
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) 

    def choose_action(self, observation, count):
        self.check_state_exist(str(observation))  

        if np.random.uniform() > self.epsilon**count: 
            state_action = self.q_table.loc[str(observation), :]  
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) 
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table.loc[s, a] 
        if not done: 
            self.check_state_exist(s_) 
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - q_predict) 


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

