from get_models import *
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class EnvModel:
    def __init__(self, name, seeds_index, data_index,  T, D):
        action_space = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                for z in range(-1, 2, 1):
                    for w in range(-1, 2, 1):
                        temp = [x / 100, y / 100, z / 100, w / 100]
                        action_space.append(temp)

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


    def step(self, state_list, action):
        action = self.action_spaces[action]

        model = state_list[0] * self.scores_cmf + state_list[1] * self.scores_nrlmf + state_list[2] * self.scores_blm + state_list[3] * self.scores_wnngip + state_list[4] * self.scores_netlaprls
        prec, rec, thr = precision_recall_curve(self.test_label, model)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(self.test_label, model)
        auc_val = auc(fpr, tpr)
        reward = aupr_val + auc_val


        state_list_ = [round(state_list[0] + action[0], 2), round(state_list[1] + action[1], 2), round(state_list[2] + action[2], 2), round(state_list[3] + action[3], 2),
                       round(state_list[4] - (action[0] + action[1] + action[2] + action[3]), 2)]


        if ((0 <= state_list_[0] <= 1) and (0 <= state_list_[1] <= 1) and
             (0 <= state_list_[2] <= 1) and (0 <= state_list_[3] <= 1) and
            (0 <= state_list_[4] <= 1)):
            model_ = state_list_[0] * self.scores_cmf + state_list_[1] * self.scores_nrlmf + state_list_[2] * self.scores_blm + state_list_[3] * self.scores_wnngip + state_list_[4] * self.scores_netlaprls
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
            Reward = -0.2
            aupr_val_ = 0
            auc_val_ = 0
            done = True
        return state_list_, Reward, done, auc_val_, aupr_val_











