TWNet for spotting the macro- and micro- intervals in the long video sequence

| File | Description | 
| --- | --- |
| requirements.txt | Install the requirements that the code needs. |
| static_pretraining.py | Static module pre-training model, using revised Fer2013dataset. |
| dynamic_pretraining.py | Dynamic module pre-training model, using processed CK+ dataset. |
| TWNet.py | Network architecture of TWNet,and the training function of TWNet. |
| utils.py | Implementation of some common methods. |
| peak_detection.py | Improved peak detection algorithm,three filters by height,distance and width. |
| casme_evaluation.py | On CAS(ME)2 data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on. |
| samm_evaluation.py | On SAMM data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on. |
| Data | Some data in this task,read the README.txt in data. |
