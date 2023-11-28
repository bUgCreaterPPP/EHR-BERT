# EHR-BERT: 

This repository provides the implementation of Bert for sequence anomaly detection. 

![alt](img/1Bert.png)

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
Create a conda environment:
```
conda create -f ./environment/environment.yml
conda activate ehr-bert
```
Reference: https://docs.conda.io/en/latest/miniconda.html

## Experiment

### Pretrain
```shell script
cd Bert

#run bert pretraining
python ehr-bert.py vocab
python ehr-bert.py train

```

### Testing
```shell script
cd Bert

#run bert prediction
python ehr-bert.py predict

```

### Folders created during execution
```shell script 
project/output //Stores intermediate files and final results during execution
```

## Paper

My research paper is under review:

```plaintext
@article{ehr-bert,
  title = {EHR-BERT: A BERT-based Model for Effective Anomaly Detection in Electronic Health Records},
  journal = {JBI},
  year = {2023},
  note = {Under Review}
}
