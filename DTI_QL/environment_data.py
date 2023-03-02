from functions import *
from get_models import *
class EnvData:
    def __init__(self, name, seeds_index, data_index):
        action_space = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                temp = [0, x / 10, y / 10]
                action_space.append(temp)
                temp = [1, x / 10, y / 10]
                action_space.append(temp)

        self.action_spaces = action_space
        self.action_number = len(self.action_spaces)
        self.reward_table = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])
        self.prec = []
        self.rec = []
        self.fpr = []
        self.tpr = []
        self.cv_data, self.X, self.D, self.T = get_dataset(name, seeds_index, data_index)

        if name == 'nr':
            self.nrn1 = np.random.normal(size=(54, 10))
            self.nrn2 = np.random.normal(size=(26, 10))
        if name == 'gpcr':
            self.nrn1 = np.random.normal(size=(223, 10))
            self.nrn2 = np.random.normal(size=(95, 10))
        if name == 'ic':
            self.nrn1 = np.random.normal(size=(210, 10))
            self.nrn2 = np.random.normal(size=(204, 10))
        if name == 'e':
            self.nrn1 = np.random.normal(size=(445, 10))
            self.nrn2 = np.random.normal(size=(664, 10))



        self.sjr, self.sir, self.sfkr, self.sud, self.sjd, self.sfkd = structure_matrix(self.cv_data, self.X, self.T, self.D)

    def memory_pool(self,  state):
        if str(state) not in self.reward_table.index:
            self.reward_table = self.reward_table.append(
                pd.Series(
                    [0] * 6,
                    index=self.reward_table.columns,
                    name=str(state),
                )
            )
            # print(reward_table)
            T = state[0] * self.sjr + state[1] * self.sir + state[2] * self.sfkr 
            D = state[3] * self.sud + state[4] * self.sjd + state[5] * self.sfkd 
            update_auc, aupr, prec, rec, fpr, tpr = auc(self.cv_data, self.X, D, T, self.nrn1, self.nrn2)

            self.reward_table.loc[str(state), 0] = update_auc
            self.reward_table.loc[str(state), 1] = aupr
            self.reward_table.loc[str(state), 2] = prec
            self.reward_table.loc[str(state), 3] = rec
            self.reward_table.loc[str(state), 4] = fpr
            self.reward_table.loc[str(state), 5] = tpr
        return self.reward_table.loc[str(state), 0], self.reward_table.loc[str(state), 1]
    def step(self, state_list, action):
        action = self.action_spaces[action]

        if action[0] == 0:
            state_list_ = [round(state_list[0]+action[1], 1), round(state_list[1]+action[2], 1),
                           round(state_list[2]-(action[1]+action[2]), 1), state_list[3], state_list[4],
                           state_list[5]]
        else:
            state_list_ = [state_list[0], state_list[1], state_list[2], round(state_list[3] + action[1], 1),
                           round(state_list[4] + action[2], 1), round(state_list[5] - (action[1] + action[2]), 1)]

        if ((0 < state_list_[0] < 1) and (0 < state_list_[1] < 1) and
            (0 < state_list_[2] < 1) and (0 < state_list_[3] < 1) and
            (0 < state_list_[4] < 1) and (0 < state_list_[5] < 1)):

            auc_cur, aupr  = self.memory_pool(state_list_)
            reward = auc_cur
            done = False
        else:
            reward = 0
            aupr = 0
            done = True
        return state_list_, reward, done, aupr











