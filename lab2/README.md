# Lab 2 - Anomaly Detection


## Dataset
Dataset for this assignment can be obtained from [here](https://itrust.sutd.edu.sg). It is about sensor data collected from a Secure Water Treatment testbed. The detail about the dataset can be seen [here](https://link.springer.com/chapter/10.1007/978-3-319-71368-7_8).

## Anomaly Detection 
In general, a similar approach as in [Goh et al.](https://ieeexplore.ieee.org/document/7911887) is used. First, a model is trained using data from the normal condition of the testbed. After that, new data is given, which could be data when anomalies occur, and the model prediction is compared with the actual sensor data. The architecture of the recurrent neural networks is also similar, but with a little bit of modification to speed up the modeling time.