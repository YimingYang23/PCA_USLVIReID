# Progressive Cross-modal Association Learning for Unsupervised Visible-Infrared Person Re-Identification

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
bash ./run_train_sysu.sh   # for SYSU-MM01
bash ./run_train_regdb.sh  # for RegDB
```
### Testing
```shell
bash ./test_sysu.sh    # for SYSU-MM01
bash ./test_regdb.sh   # for RegDB
```

### Performance on SYSU-MM01 (Batchsize = 64)
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="5">All Search</th>
    <th class="tg-baqh" colspan="5">Indoor Search</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">Rank-1</td>
    <td class="tg-baqh">Rank-10</td>
    <td class="tg-baqh">Rank-20</td>
    <td class="tg-baqh">mAP</td>
    <td class="tg-baqh">mINP</td>
    <td class="tg-baqh">Rank-1</td>
    <td class="tg-baqh">Rank-10</td>
    <td class="tg-baqh">Rank-20</td>
    <td class="tg-baqh">mAP</td>
    <td class="tg-baqh">mINP</td>
  </tr>
  <tr>
    <td class="tg-baqh">54.39</td>
    <td class="tg-baqh">89.09</td>
    <td class="tg-baqh">95.62</td>
    <td class="tg-baqh">51.95</td>
    <td class="tg-baqh">38.09</td>
    <td class="tg-baqh">59.69</td>
    <td class="tg-baqh">93.90</td>
    <td class="tg-baqh">98.14</td>
    <td class="tg-baqh">66.72</td>
    <td class="tg-baqh">62.44</td>
  </tr>
</tbody>
</table>
