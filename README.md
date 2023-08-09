# PCA: Progressive Cross-modal Association Learning for Unsupervised Visible-Infrared Person Re-Identification

# Dataset
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare_sysu.py and prepare_regdb.py to convert the dataset format.

## Prerequisites
- Linux
- A minimum of 24GB of GPU memory (Batchsize = 64)

## Getting Started
git clone https://github.com/YimingYang23/PCA_USLVIReID.git

### install dependencies
- conda create -n PCAReID python==3.7
- conda activate PCAReID
- pip install -r requirement.txt

### Training
```shell
./PCA_sysu.sh   # for SYSU-MM01
./PCA_regdb.sh  # for RegDB
```
### Testing
```shell
./test_sysu.sh    # for SYSU-MM01
./test_regdb.sh   # for RegDB
```
