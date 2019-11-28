# deviation-network
Deep anomaly detection with deviation networks (KDD19)

Guansong Pang, Chunhua Shen, Anton van den Hengel, University of Adelaide, Adelaide, Australia

Deviation network (DevNet) is introduced in our KDD19 paper, which leverages a limited number of labeled anomaly data and a large set of unlabeled data to perform end-to-end anomaly score learning. It addresses a weakly supervised anomaly detection problem in that the anomalies are partially observed only and we have no labeled normal data.

Unlike other deep anomaly detection methods that focus on using data reconstruction as the driving force to learn new representations, DevNet is devised to learn the anomaly scores directly. Therefore, DevNet directly optimize the anomaly scores, whereas most of current deep anomaly detection methods optimize the feature representations. The resulting DevNet model achieves significantly better anomaly scoring than the competing deep methods. Also, due to the end-to-end anomaly scoring, DevNet can also exploit the labeled anomaly data much more effectively. 

The full paper can be found at https://dl.acm.org/citation.cfm?id=3330871 or https://arxiv.org/abs/1911.08623
