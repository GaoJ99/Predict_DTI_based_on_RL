'''
[1] Mei, Jian-Ping, et al. "Drug target interaction prediction by learning from local information and neighbors." Bioinformatics 29.2 (2013): 238-245.
[2] van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori. "Gaussian interaction profile kernels for predicting drug-target interaction." Bioinformatics 27.21 (2011): 3036-3043.

Default Parameters:
    alpha = 0.5
    gamma = 1.0 (the gamma0 in [1], see Eq. 11 and 12 for details)
    avg = False (True: g=mean, False: g=max)
    sigma = 1.0 (The regularization parameter used for the RLS-avg classifier)
Please refer to Section 4.1 in [1] and and Section 4 in [2] for the details.
'''
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class BLMNII:

    def __init__(self, alpha=0.5, gamma=1.0, sigma=1.0, avg=False):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        if avg in ('false', 'False', False):
            self.avg = False
        if avg in ('true', 'True', True):
            self.avg = True

    def kernel_combination(self, R, S, new_inx, bandwidth):
        K = self.alpha*S+(1.0-self.alpha)*rbf_kernel(R, gamma=bandwidth)
        K[new_inx, :] = S[new_inx, :]
        K[:, new_inx] = S[:, new_inx]
        return K

    def rls_train(self, R, S, K, train_inx, new_inx):
        Y = R.copy()
        for d in new_inx:
            Y[d, :] = np.dot(S[d, train_inx], Y[train_inx, :])
            x1, x2 = np.max(Y[d, :]), np.min(Y[d, :])
            Y[d, :] = (Y[d, :]-x2)/(x1-x2)
        vec = np.linalg.inv(K+self.sigma*np.eye(K.shape[0]))
        return np.dot(np.dot(K, vec), Y)

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        m, n = intMat.shape
        x, y = np.where(R > 0)
        drugMat = (drugMat+drugMat.T)/2
        targetMat = (targetMat+targetMat.T)/2
        train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
        train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
        new_drugs = np.array(list(set(range(m)) - set(x.tolist())), dtype=np.int32)
        new_targets = np.array(list(set(range(n)) - set(y.tolist())), dtype=np.int32)
        drug_bw = self.gamma*m/len(x)
        target_bw = self.gamma*n/len(x)

        Kd = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
        Kt = self.kernel_combination(R.T, targetMat, new_targets, target_bw)
        self.Y1 = self.rls_train(R, drugMat, Kd, train_drugs, new_drugs)
        self.Y2 = self.rls_train(R.T, targetMat, Kt, train_targets, new_targets)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        x, y = inx[:, 0], inx[:, 1]
        if self.avg:
            scores = 0.5*(self.Y1[x, y]+self.Y2.T[x, y])
        else:
            scores = np.maximum(self.Y1[x, y], self.Y2.T[x, y])
        return scores

    def evaluation(self, test_data, test_label):
        x, y = test_data[:, 0], test_data[:, 1]
        if self.avg:
            scores = 0.5*(self.Y1[x, y]+self.Y2.T[x, y])
        else:
            scores = np.maximum(self.Y1[x, y], self.Y2.T[x, y])
        # prec, rec, thr = precision_recall_curve(test_label, scores)
        # aupr_val = auc(rec, prec)
        # fpr, tpr, thr = roc_curve(test_label, scores)
        # auc_val = auc(fpr, tpr)
        return scores

    def __str__(self):
        return "Model:BLMNII, alpha:%s, gamma:%s, sigma:%s, avg:%s" % (self.alpha, self.gamma, self.sigma, self.avg)


