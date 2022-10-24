# AD-TFM-AT-Model  
Detection of incipient faults in power distribution network (LSTM/Adaptive Wavelet trasnform/Attention).
## Requirement  
* Python 3.6  
* Tensorflow-gpu 1.14.0  
* Keras 2.25  
## Train and test the model  
Run command below to train and test the model:  

```python test_tf.py```  

## Experiment  
Datasets are obtained from a small Incipient Fault dataset in Power Distribution (IFPD) system from [1] (https://dx.doi.org/10.21227/bwjy-7e05), and a relatively large dataset logged by State Grid Corporation of China in AnHui Province (SGAH) from [2] (https://github.com/smartlab-hfut/SGAH-datasets.git).  


   device: Tesla V100   
   dataset: IFPD and SGAH   
   optimizer: Adam(lr=0.001, eps=1e-08)  
   batch：800 


These are the result for the incipient fault detection in two datasets.

| Metrics | Accuracy | Precision | Recall | F1score |
| ------- |:---:| :--:| :--: | :--: |
| IFPD | 0.97 | 0.97 | 0.96 | 0.96 |
| SGAH | 0.99 | 0.97 | 0.98 | 0.98 |

![evaluate1](/figures/IFPD.png)  
Fig.1 ROC of AD-TFM-AT model on IFPD.  

![evaluate2](/figures/SGAH.png)  
Fig.2 ROC of AD-TFM-AT model on SGAH.

## Reference
If you use the codes or the datasets, please cite the following papers:   
   ```luohuan  
   is  
   is```  
            
               
## Copyright  
See [LICENSE](LICENSE) for details.










































