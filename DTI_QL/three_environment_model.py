from get_models import *
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class EnvModel:
    def __init__(self, name, seeds_index, data_index,  T, D):
    # def __init__(self, name):
        # 动作空间
        action_space = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                temp = [x / 100, y / 100]
                action_space.append(temp)

        self.models = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4],
                       [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

        self.action_spaces = action_space
        self.action_number = len(self.action_spaces)
        self.prec = []
        self.rec = []
        self.fpr = []
        self.tpr = []

        self.scores_cmf, self.test_label = np.array(get_cmf_model(name, seeds_index, data_index, T, D))
        self.scores_nrlmf, self.test_label = np.array(get_nrlmf_model(name, seeds_index, data_index, T, D))
        self.scores_blm, self.test_label = np.array(get_blm_model(name, seeds_index, data_index, T, D))
        self.scores_wnngip, self.test_label = np.array(get_wnngip_model(name, seeds_index, data_index, T, D))
        self.scores_netlaprls, self.test_label = np.array(get_netlaprls_model(name, seeds_index, data_index, T, D))
        self.scores_models = [self.scores_cmf, self.scores_nrlmf, self.scores_blm, self.scores_wnngip, self.scores_netlaprls]

    def choose(self, state_list, count):
        choose_index = self.models[count]
        model = state_list[0] * self.scores_models[choose_index[0]] + state_list[1] * self.scores_models[choose_index[1]] + state_list[2] * self.scores_models[choose_index[2]]
        return model


    # 根据状态和动作得出即时奖励和下一个状态以及是否结束当前迭代
    def step(self, state_list, action, count):
        action = self.action_spaces[action]

        model = self.choose(state_list, count)

        prec, rec, thr = precision_recall_curve(self.test_label, model)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(self.test_label, model)
        auc_val = auc(fpr, tpr)
        reward = aupr_val + auc_val

        # 根据所采取得动作来更新以下状态
        state_list_ = [round(state_list[0] + action[0], 2), round(state_list[1] + action[1], 2), round(state_list[2] - action[0] - action[1], 2)]


        #  判断更新后的状态是否符合现实
        if ((0 <= state_list_[0] <= 1) and (0 <= state_list_[1] <= 1) and (0 <= state_list_[2] <= 1)):
            # 算出合并的模型
            model_ = self.choose(state_list_, count)
            prec_, rec_, thr_ = precision_recall_curve(self.test_label, model_)
            self.rec = rec_
            self.prec = prec_
            aupr_val_ = auc(rec_, prec_)
            fpr_, tpr_, thr_ = roc_curve(self.test_label, model_)
            self.fpr = fpr_
            self.tpr = tpr_
            auc_val_ = auc(fpr_, tpr_)
            reward_ = aupr_val_ + auc_val_
            if reward_ > reward:
                Reward = 0.1
            if reward_ == reward:
                Reward = 0
            if reward_ < reward:
                Reward = -0.1
            done = False
        else:
            Reward = -5
            aupr_val_ = 0
            auc_val_ = 0
            done = True
        # print("state_list_:", state_list_)
        return state_list_, Reward, done, auc_val_, aupr_val_

#
if __name__ == "__main__":
    env = EnvModel('nr')
    print("env.action_spaces:", env.action_spaces)
    # print("env.action_number", env.action_number)
    # for i in range(env.action_number):
    #     print(i, env.action_spaces[i])
    # print(env.models)











