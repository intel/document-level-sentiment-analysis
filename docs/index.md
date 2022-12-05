# Welcome to DLSA Pages

DLSA is Intel optimized representative End-to-end Fine-Tuning & Inference pipeline for Document level sentiment analysis using BERT model implemented with Hugging face transformer API.

![Image](assets/images/DLSA_workflow.PNG)

## Prerequisites
### Download the repo

```
#download the repo
git clone https://github.com/intel/document-level-sentiment-analysis.git
cd frameworks.ai.end2end-ai-pipelines.dlsa/profiling-transformers
git checkout v1.0.0
```

### Download the datasets:

```
mkdir datasets
cd datasets
#download and extract SST-2 dataset
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip && unzip SST-2.zip && mv SST-2 sst
#download and extract IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && tar -zxf aclImdb_v1.tar.gz
```
>Note: Make sure the network connections work well for downloading the datasets.

## Deploy the test environment 
### Download Miniconda and install it

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

> Note: If you have already installed conda on your system, just skip this step.

### Prepare the conda environment for DLSA

```
conda create -n dlsa python=3.8 --yes
conda activate dlsa
sh install.sh
```

## Running DLSA Inference Pipeline

| Implementations                                          | Model    | API         | Framework      | Precision      |
| -------------------------------------------------------- | -------- | ----------- | -------------- | -------------- |
| [Run with HF Transformers](inference/hf-transformers.md) | HF Model | Trainer     | PyTorch + IPEX | FP32,BF16      |
| [Run with Stock Pytorch](inference/stock-pytorch.md)     | HF Mode  | Non-trainer | PyTorch        | FP32           |
| [Run with IPEX](inference/ipex.md)                       | HF Mode  | Non-trainer | PyTorch + IPEX | FP32,BF16,INT8 |

## Running DLSA Fine-Tuning Pipeline

### Single Node Fine-Tuning

|  Implementations                               | Model    | Instance | API         | Framework       | Precision  |
| ---------------------------------- | -------- | -------- | ----------- | ----------------------- | ---------- |
| [Run with HF Transformers + IPEX ](fine-tuning/single-node-trainer.md)   | HF Model | Single   | Trainer     | PyTorch + IPEX          | FP32, BF16 |
| [Run with Stock Pytorch](fine-tuning/single-node-stock-pytorch.md) | HF Model  | Single   | Non-trainer | PyTorch                 | FP32       |
| [Run with IPEX (Single Instance)](fine-tuning/single-node-ipex.md) | HF Model  | Single   | Non-trainer | PyTorch + IPEX          | FP32,BF16  |
| [Run with IPEX (Multi Instance)](fine-tuning/multi-nodes-ipex.md) | HF Model  | Multiple | Non-trainer | PyTorch + IPEX          | FP32,BF16  |


## Issue Tracking 
E2E DLSA tracks both bugs and enhancement requests using [Github](https://github.com/intel/document-level-sentiment-analysis/issues). We welcome input, however, before filing a request, please make sure you do the following:
Search the Github issue database.
