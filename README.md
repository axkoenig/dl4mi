# dl4mi

This work compares three deep learning approaches for COVID-19 detection from chest radiographies. Our first method performs transfer learning with a pre-trained ResNet-50. Further, we propose a model that relies on anomaly detection with the U-Net architecture and a ResNet-50 classifier. Our third approach is based on multitask learning where a modified U-Net performs a reconstruction task and a classification task simultaneously.

## Setup Instructions

Create virtual environment and follow these steps.

```bash
git clone git@github.com:axkoenig/dl4mi.git
pip install -e dl4mi
```

Download the COVIDx3 dataset presented in the [COVID-Net paper](https://arxiv.org/abs/2003.09871). Please follow instructions in the [official repository](https://github.com/lindawangg/COVID-Net) to download the dataset. Place the data in a directory of your choice. The subdirectories of your data directory should be called "train" and "test". Place the respective files in these directories. 

The models can be trained using the ```classifier.py```script in the respective directories. To check what arguments can be specified run the script with the ```-h``` flag. 

```bash
python 3_multitask_learning/classifier.py -h
```