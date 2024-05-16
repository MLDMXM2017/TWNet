TWNet for spotting the macro- and micro- intervals in the long video sequence

#########
1.requirements.txt: Install the requirements that the code needs
2.static_pretraining.py: the static module pre-training model, using revised Fer2013dataset
3.dynamic_pretraining.py: the dynamic module pre-training model, using processed CK+ dataset
4.TWNet.py: the Network architecture of TWNet,and the training function of TWNet
5.utils.py: Implementation of some common methods
6.peak_detection.py: the improved peak detection algorithm,three filters by height,distance and width
7.casme_evaluation.py: On CAS(ME)2 data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on
8.samm_evaluation.py:  On SAMM data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on
9.data: some data in this task,read the README.txt in data
#########