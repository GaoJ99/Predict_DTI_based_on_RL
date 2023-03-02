from environment_data import EnvData
from environment_model import EnvModel
from rl_Qtable import QLearningTable
from functions import *
import pandas as pd
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def model_data():
    EPISDOE = 5000 
    STEP = 50
    experimental_results = []
    aucs = []
    max = 0
    count = 0
    final_state = list()

    for episode in range(EPISDOE):
        state_data = [0.3, 0.3, 0.4, 0.3, 0.3, 0.4]
        episode_auc = []
        step = 0
        result_each_round = []
        while True:
            step += 1
            experimental_result = []
            action = RL_data.choose_action(state_data, count)
            observation_, reward, done, aupr = env1.step(state_data, action)
            if episode < 5:
                print(str(episode)+str('/')+str(step), 'state_list:', state_data, 'observation_:', observation_, 'action:', action, 'auc:', reward, 'aupr:', aupr)
            add_data_to_list1(experimental_result, experimental_results, episode, step, state_data, observation_, reward)
            # RL.store_memory(state_list, action, reward, observation_, done)  # (self, s, a, r, s_, d)
            RL_data.learn(str(state_data), action, reward, str(observation_), done)
            if max <= reward:
                max = reward
                final_state = observation_
            state_data = observation_
            episode_auc.append(reward)

            if done:
                aucs.append(episode_auc[:-1])
                break

            if step == STEP:
                aucs.append(episode_auc)
                break
            count += 1

        print("第"+str(episode)+"轮：", "步数："+str(step), "权重分配："+str(observation_), "auc：", reward, 'aupr:', aupr)
        add_data_to_list2(result_each_round, experimental_results, episode, step, observation_, reward, aupr)

    T = final_state[0] * env1.sjr + final_state[1] * env1.sir + final_state[2] * env1.sfkr
    D = final_state[3] * env1.sud + final_state[4] * env1.sjd + final_state[5] * env1.sfkd
    print("final_state", final_state)
    return T, D

def model(name, data_index, model_name):

    aupr_rec_name = 'aupr_rec'
    aupr_prec_name = 'aupr_prec'
    auc_fpr_name = 'auc_fpr'
    auc_tpr_name = 'auc_tpr'
    # EPISDOE = 5000 
    EPISDOE = 1 
    # STEP = 100
    STEP = 1
    Reward_list = []
    f = open('Result/five_model/Data.csv', 'w', newline="")

    writer = csv.writer(f)

    for episode in range(EPISDOE):
        if model_name == 'cmf':
            state_list = [1, 0, 0, 0, 0]   # cmf
        if model_name == 'nrlmf':
            state_list = [0, 1, 0, 0, 0]   # nrlmf
        if model_name == 'blm':
            state_list = [0, 0, 1, 0, 0]   # blm
        if model_name == 'wnngip':
            state_list = [0, 0, 0, 1, 0]   # wnngip
        if model_name == 'netlaprls':
            state_list = [0, 0, 0, 0, 1]   # netlaprls
        step = 1
        Reward = 0
        while True:
            action = 40
            observation_, reward, done, auc, aupr = env2.step(state_list, action)
            state_list = observation_
            Reward += reward
            if done or step == STEP:
                Reward_list.append(Reward/step)
                break
            step += 1

        print("第"+str(episode)+"轮：", "步数："+str(step), "权重分配："+str(observation_), "总Reward：", Reward, 'auc:', auc, 'aupr:', aupr)
        
    data = pd.DataFrame(Reward_list)
    save_data_to_csv(data, 'average_rewards_per_round', name, data_index)

    aupr_rec = pd.DataFrame(env2.rec)
    save_data_to_csv(aupr_rec, aupr_rec_name, name, data_index)
    aupr_prec = pd.DataFrame(env2.prec)
    save_data_to_csv(aupr_prec, aupr_prec_name, name, data_index)

    auc_fpr = pd.DataFrame(env2.fpr)
    save_data_to_csv(auc_fpr, auc_fpr_name, name, data_index)
    auc_tpr = pd.DataFrame(env2.tpr)
    save_data_to_csv(auc_tpr, auc_tpr_name, name, data_index)
    f.close()

    x = [x for x in range(1, len(Reward_list) + 1)]
    y = Reward_list
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    # [0, 4]
    data_index = 0

    seeds_index = 0
    name = 'e'   # ['nr', 'gpcr', 'ic', 'e']
    gamma = 0.9
    learning_rate = 0.1
    epsilon1 = 0.99997

    env1 = EnvData(name, seeds_index, data_index)
    RL_data = QLearningTable(list(range(env1.action_number)), env1.action_spaces, learning_rate, gamma, epsilon1)
    print("数据开始融合......")
    T, D = model_data()


    epsilon2 = 0.9992
    model_name = 'netlaprls'  # [cmf, nrlmf, blm, wnngip, netlaprls]
    env2 = EnvModel(name, seeds_index, data_index, T, D)
    RL_model = QLearningTable(list(range(env2.action_number)), env2.action_spaces, learning_rate, gamma, epsilon2)
    print("模型开始融合......")
    model(name, data_index, model_name)
