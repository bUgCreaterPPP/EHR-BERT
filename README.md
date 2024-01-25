# EHR-BERT: Anomaly Detection in Electronic Health Records using BERT

This repository provides the implementation of Bert for sequence anomaly detection. 

![alt](img/1Bert.png)

## Prerequisites

Before using this code, make sure you have the following prerequisites installed on your system:

- Ubuntu 20.04
- NVIDIA driver 460.73.01
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation

To set up the required environment, follow these steps:

1. Clone the repository.
2. Create a Conda environment using the provided YAML file:

   ```shell script
   conda env create -f ./environment/environment.yml
   conda activate ehr-bert
    ```

Reference: https://docs.conda.io/en/latest/miniconda.html

## Running Experiments with Sample Data

We provide preprocessed synthetic sequence data files for testing in [example](https://github.com/hniu1/EHR-BERT/tree/main/output/example).

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
  year = {2024},
  note = {Under Review}
}
