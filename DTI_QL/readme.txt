Overview：
This repository provides the datasets, method code, and model code used by the QLDTI(Predicting Drug-Target Interaction based on Q-learning) method.
All code is written in Python 3 and it mainly uses libraries such as sklearn,pandas,numpy.

Usage:
1.the datasets folder contains the four benchmark datasets required,the dataset folder contains the four benchmark datasets required, namely NR, GPCR, IC, and E.
2.blm.py, cmf.py, nrlmf.py, wnngip.py and netlaprls.py, these five .py files are the codes for each of the five models.
3.calculate_similarity_matrix.py includes methods for calculating the similarity between drugs and targets.
4.rl_Qtable.py is the code for establishing the Q table.
5.run_dtp2.py is the main code for building the QLDTI method.By running this file and using the QLDTI method to predict DTI, the average reward convergence figure, AUC value, and AUPR value will be obtained.
