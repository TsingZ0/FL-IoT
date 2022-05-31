# FL-IoT
This is a platform containing the datasets and federated learning algorithms in IoT environments. Except for the datasets and algorithms, the features of this platform are the same as my another federated learning platform [PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID). 


## Algorithms (updating)
- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) *AISTATS 2017*
- **FedPer** — [Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818.pdf)
- **FedProx** — [Federated Optimization for Heterogeneous Networks](https://openreview.net/pdf?id=SkgwE5Ss3N) *ICLR 2020*
- **APFL** — [Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461.pdf)
- **FedAvg-FT** — [Federated Evaluation of On-device Personalization](https://arxiv.org/pdf/1910.10252.pdf)
- **FedProx-FT** — [Federated Evaluation of On-device Personalization](https://arxiv.org/pdf/1910.10252.pdf)
- **FedHome** — [FedHome: Cloud-Edge based Personalized Federated Learning for In-Home Health Monitoring](https://ieeexplore.ieee.org/abstract/document/9296274?casa_token=TlOJ9dKfU0IAAAAA:1Pt_76jE1d0OcbREzAatozlHmfCoxuNNb7tAgsezomR4kr1u564j4KW19TzxPagB64_MoNE) *IEEE Transactions on Mobile Computing*
- **FedPrune** — [FedPrune: Personalized and Communication-Efficient Federated Learning on Non-IID Data](https://link.springer.com/chapter/10.1007/978-3-030-92307-5_50) *ICONIP 2021*


## Datasets (updating)
Two public datasets are used: [HAR (Human Activity Recognition Using Smartphones Data Set)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and [PAMAP2 (Physical Activity Monitoring Data Set)](http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring). Both of them are collected using sensors in real-world settings, so the data naturally belongs to each subject (i.e. client). For the detailed descriptions of these two datasets, please visit the given URLs. HAR has been pre-processed before download, but PAMAP2 just contains raw data. Thus, I pre-process PAMAP2 following the method for HAR. Specifically, (1) I only keep the IMU (Inertial Measurement Unit) data; (2) I sample the signals in fixed-width sliding windows of 2.56 sec and 50% overlap (256 readings/window). 

Although these datasets can be used for various tasks, I only condister the classification task here. HAR contains 30 clients with data in 6 classes and PAMAP2 contains 9 clients with data in 12 classes. Note that I do not shuffle the data, as the data is collected over time. 

In both HAR and PAMAP2, the data among clients are heterogeneous (feature shift). As shown in the following, PAMAP2 is more heterogeneous than HAR. 

### Dataset generating examples
The output of `generate_har.py`
```
Client 0         Size of data: 347       Labels:  [0 1 2 3 4 5]
                 Samples of labels:  [(0, 95), (1, 53), (2, 49), (3, 47), (4, 53), (5, 50)]
--------------------------------------------------
Client 1         Size of data: 302       Labels:  [0 1 2 3 4 5]
                 Samples of labels:  [(0, 59), (1, 48), (2, 47), (3, 46), (4, 54), (5, 48)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>
    Client 2         Size of data: 341       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 58), (1, 59), (2, 49), (3, 52), (4, 61), (5, 62)]
    --------------------------------------------------
    Client 3         Size of data: 317       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 60), (1, 52), (2, 45), (3, 50), (4, 56), (5, 54)]
    --------------------------------------------------
    Client 4         Size of data: 302       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 56), (1, 47), (2, 47), (3, 44), (4, 56), (5, 52)]
    --------------------------------------------------
    Client 5         Size of data: 325       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 57), (1, 51), (2, 48), (3, 55), (4, 57), (5, 57)]
    --------------------------------------------------
    Client 6         Size of data: 308       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 57), (1, 51), (2, 47), (3, 48), (4, 53), (5, 52)]
    --------------------------------------------------
    Client 7         Size of data: 281       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 48), (1, 41), (2, 38), (3, 46), (4, 54), (5, 54)]
    --------------------------------------------------
    Client 8         Size of data: 288       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 52), (1, 49), (2, 42), (3, 50), (4, 45), (5, 50)]
    --------------------------------------------------
    Client 9         Size of data: 294       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 53), (1, 47), (2, 38), (3, 54), (4, 44), (5, 58)]
    --------------------------------------------------
    Client 10        Size of data: 316       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 59), (1, 54), (2, 46), (3, 53), (4, 47), (5, 57)]
    --------------------------------------------------
    Client 11        Size of data: 320       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 50), (1, 52), (2, 46), (3, 51), (4, 61), (5, 60)]
    --------------------------------------------------
    Client 12        Size of data: 327       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 57), (1, 55), (2, 47), (3, 49), (4, 57), (5, 62)]
    --------------------------------------------------
    Client 13        Size of data: 323       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 59), (1, 54), (2, 45), (3, 54), (4, 60), (5, 51)]
    --------------------------------------------------
    Client 14        Size of data: 328       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 54), (1, 48), (2, 42), (3, 59), (4, 53), (5, 72)]
    --------------------------------------------------
    Client 15        Size of data: 366       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 51), (1, 51), (2, 47), (3, 69), (4, 78), (5, 70)]
    --------------------------------------------------
    Client 16        Size of data: 368       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 61), (1, 48), (2, 46), (3, 64), (4, 78), (5, 71)]
    --------------------------------------------------
    Client 17        Size of data: 364       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 56), (1, 58), (2, 55), (3, 57), (4, 73), (5, 65)]
    --------------------------------------------------
    Client 18        Size of data: 360       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 52), (1, 40), (2, 39), (3, 73), (4, 73), (5, 83)]
    --------------------------------------------------
    Client 19        Size of data: 354       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 51), (1, 51), (2, 45), (3, 66), (4, 73), (5, 68)]
    --------------------------------------------------
    Client 20        Size of data: 408       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 52), (1, 47), (2, 45), (3, 85), (4, 89), (5, 90)]
    --------------------------------------------------
    Client 21        Size of data: 321       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 46), (1, 42), (2, 36), (3, 62), (4, 63), (5, 72)]
    --------------------------------------------------
    Client 22        Size of data: 372       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 59), (1, 51), (2, 54), (3, 68), (4, 68), (5, 72)]
    --------------------------------------------------
    Client 23        Size of data: 381       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 58), (1, 59), (2, 55), (3, 68), (4, 69), (5, 72)]
    --------------------------------------------------
    Client 24        Size of data: 409       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 74), (1, 65), (2, 58), (3, 65), (4, 74), (5, 73)]
    --------------------------------------------------
    Client 25        Size of data: 392       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 59), (1, 55), (2, 50), (3, 78), (4, 74), (5, 76)]
    --------------------------------------------------
    Client 26        Size of data: 376       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 57), (1, 51), (2, 44), (3, 70), (4, 80), (5, 74)]
    --------------------------------------------------
    Client 27        Size of data: 382       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 54), (1, 51), (2, 46), (3, 72), (4, 79), (5, 80)]
    --------------------------------------------------
    Client 28        Size of data: 344       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 53), (1, 49), (2, 48), (3, 60), (4, 65), (5, 69)]
    --------------------------------------------------
    Client 29        Size of data: 383       Labels:  [0 1 2 3 4 5]
                    Samples of labels:  [(0, 65), (1, 65), (2, 62), (3, 62), (4, 59), (5, 70)]
    --------------------------------------------------
    Total number of samples: 10299
    The number of train samples: [260, 226, 255, 237, 226, 243, 231, 210, 216, 220, 237, 240, 245, 242, 246, 274, 276, 273, 270, 265, 306, 240, 279, 285, 306, 294, 282, 286, 258, 287]
    The number of test samples: [87, 76, 86, 80, 76, 82, 77, 71, 72, 74, 79, 80, 82, 81, 82, 92, 92, 91, 90, 89, 102, 81, 93, 96, 103, 98, 94, 96, 86, 96]

    Saving to disk.

    Finish generating dataset.
</details>
<br/>

The output of `generate_pamap2.py`
```
Client 0         Size of data: 1932      Labels:  [ 0  1  2  3  4  5  6  7  8  9 10 11]
                 Samples of labels:  [(0, 99), (1, 213), (2, 183), (3, 168), (4, 171), (5, 164), (6, 182), (7, 157), (8, 122), (9, 113), (10, 178), (11, 182)]
--------------------------------------------------
Client 1         Size of data: 2031      Labels:  [ 0  1  2  3  4  5  6  7  8  9 10 11]
                 Samples of labels:  [(0, 102), (1, 181), (2, 173), (3, 197), (4, 252), (5, 70), (6, 194), (7, 231), (8, 133), (9, 115), (10, 160), (11, 223)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>
    Client 2         Size of data: 1348      Labels:  [ 1  2  3  4  8  9 10 11]
                    Samples of labels:  [(1, 170), (2, 225), (3, 160), (4, 225), (8, 81), (9, 113), (10, 157), (11, 217)]
    --------------------------------------------------
    Client 3         Size of data: 1788      Labels:  [ 1  2  3  4  6  7  8  9 10 11]
                    Samples of labels:  [(1, 178), (2, 199), (3, 193), (4, 247), (6, 175), (7, 213), (8, 128), (9, 107), (10, 155), (11, 193)]
    --------------------------------------------------
    Client 4         Size of data: 2108      Labels:  [ 0  1  2  3  4  5  6  7  8  9 10 11]
                    Samples of labels:  [(0, 58), (1, 183), (2, 210), (3, 173), (4, 248), (5, 191), (6, 190), (7, 204), (8, 110), (9, 96), (10, 189), (11, 256)]
    --------------------------------------------------
    Client 5         Size of data: 1932      Labels:  [ 1  2  3  4  5  6  7  8  9 10 11]
                    Samples of labels:  [(1, 181), (2, 180), (3, 190), (4, 198), (5, 176), (6, 158), (7, 207), (8, 101), (9, 85), (10, 163), (11, 293)]
    --------------------------------------------------
    Client 6         Size of data: 1798      Labels:  [ 1  2  3  4  5  6  7  8  9 10 11]
                    Samples of labels:  [(1, 198), (2, 94), (3, 201), (4, 262), (5, 26), (6, 176), (7, 222), (8, 137), (9, 86), (10, 167), (11, 229)]
    --------------------------------------------------
    Client 7         Size of data: 2027      Labels:  [ 0  1  2  3  4  5  6  7  8  9 10 11]
                    Samples of labels:  [(0, 67), (1, 186), (2, 180), (3, 196), (4, 245), (5, 127), (6, 197), (7, 223), (8, 91), (9, 71), (10, 188), (11, 256)]
    --------------------------------------------------
    Client 8         Size of data: 48        Labels:  [0]
                    Samples of labels:  [(0, 48)]
    --------------------------------------------------
    Total number of samples: 15012
    The number of train samples: [1449, 1523, 1011, 1341, 1581, 1449, 1348, 1520, 36]
    The number of test samples: [483, 508, 337, 447, 527, 483, 450, 507, 12]

    Saving to disk.

    Finish generating dataset.
</details>
<br/>


## Models
I only use a CNN for HAR and PAMAP2 here. 


## How to start simulating 
Please refer to `har.sh` and `pamap.sh`.