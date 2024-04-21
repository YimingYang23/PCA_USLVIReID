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
sh ./run_train_sysu.sh   # for SYSU-MM01
sh ./run_train_regdb.sh  # for RegDB
```
### Testing
```shell
sh ./test_sysu.sh    # for SYSU-MM01
sh ./test_regdb.sh   # for RegDB
```
## Performance
We conducted multiple tests of our method to achieve the best results.
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


### Performance on RegDB (10 trials, Batchsize = 64)
<table class="tg">
<thead>
  <tr>
    <th class="tg-gaoc" rowspan="2">Trial</th>
    <th class="tg-gaoc" colspan="5">Visible to Infrared</th>
    <th class="tg-gaoc" colspan="5">Infrared to Visible</th>
  </tr>
  <tr>
    <th class="tg-gaoc">Rank-1</th>
    <th class="tg-gaoc">Rank-10</th>
    <th class="tg-gaoc">Rank-20</th>
    <th class="tg-gaoc">mAP</th>
    <th class="tg-gaoc">mINP</th>
    <th class="tg-gaoc">Rank-1</th>
    <th class="tg-gaoc">Rank-10</th>
    <th class="tg-gaoc">Rank-20</th>
    <th class="tg-gaoc">mAP</th>
    <th class="tg-gaoc">mINP</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-gaoc">1</td>
    <td class="tg-3ttq">88.83</td>
    <td class="tg-gaoc">95.97</td>
    <td class="tg-gaoc">97.67</td>
    <td class="tg-gaoc">84.86</td>
    <td class="tg-gaoc">75.42</td>
    <td class="tg-3ttq">86.99</td>
    <td class="tg-gaoc">96.26</td>
    <td class="tg-gaoc">97.86</td>
    <td class="tg-gaoc">82.21</td>
    <td class="tg-gaoc">70.99</td>
  </tr>
  <tr>
    <td class="tg-s4h7">2</td>
    <td class="tg-0udb">86.41</td>
    <td class="tg-s4h7">94.03</td>
    <td class="tg-s4h7">95.83</td>
    <td class="tg-s4h7">82.95</td>
    <td class="tg-s4h7">72.97</td>
    <td class="tg-0udb">88.25</td>
    <td class="tg-s4h7">95.10</td>
    <td class="tg-s4h7">96.26</td>
    <td class="tg-s4h7">83.02</td>
    <td class="tg-s4h7">70.57</td>
  </tr>
  <tr>
    <td class="tg-s4h7">3</td>
    <td class="tg-0udb">84.76</td>
    <td class="tg-s4h7">93.36</td>
    <td class="tg-s4h7">95.63</td>
    <td class="tg-s4h7">80.81</td>
    <td class="tg-s4h7">71.10</td>
    <td class="tg-0udb">83.93</td>
    <td class="tg-s4h7">92.67</td>
    <td class="tg-s4h7">94.51</td>
    <td class="tg-s4h7">79.36</td>
    <td class="tg-s4h7">66.78</td>
  </tr>
  <tr>
    <td class="tg-s4h7">4</td>
    <td class="tg-0udb">85.92</td>
    <td class="tg-s4h7">93.88</td>
    <td class="tg-s4h7">96.36</td>
    <td class="tg-s4h7">82.32</td>
    <td class="tg-s4h7">72.85</td>
    <td class="tg-0udb">86.17</td>
    <td class="tg-s4h7">94.03</td>
    <td class="tg-s4h7">95.68</td>
    <td class="tg-s4h7">81.23</td>
    <td class="tg-s4h7">69.01</td>
  </tr>
  <tr>
    <td class="tg-s4h7">5</td>
    <td class="tg-hi9g">85.29</td>
    <td class="tg-4jb6">94.22</td>
    <td class="tg-4jb6">96.60</td>
    <td class="tg-4jb6">81.29</td>
    <td class="tg-4jb6">70.52</td>
    <td class="tg-hi9g">85.44</td>
    <td class="tg-4jb6">93.83</td>
    <td class="tg-4jb6">96.26</td>
    <td class="tg-4jb6">80.02</td>
    <td class="tg-4jb6">66.13</td>
  </tr>
  <tr>
    <td class="tg-s4h7">6</td>
    <td class="tg-0udb">87.09</td>
    <td class="tg-s4h7">95.39</td>
    <td class="tg-s4h7">97.09</td>
    <td class="tg-s4h7">83.01</td>
    <td class="tg-s4h7">72.08</td>
    <td class="tg-0udb">86.84</td>
    <td class="tg-s4h7">94.22</td>
    <td class="tg-s4h7">97.04</td>
    <td class="tg-s4h7">81.84</td>
    <td class="tg-s4h7">69.57</td>
  </tr>
  <tr>
    <td class="tg-s4h7">7</td>
    <td class="tg-0udb">85.05</td>
    <td class="tg-s4h7">93.06</td>
    <td class="tg-s4h7">95.05</td>
    <td class="tg-s4h7">81.01</td>
    <td class="tg-s4h7">70.32</td>
    <td class="tg-0udb">84.37</td>
    <td class="tg-s4h7">93.79</td>
    <td class="tg-s4h7">96.07</td>
    <td class="tg-s4h7">79.03</td>
    <td class="tg-s4h7">65.78</td>
  </tr>
  <tr>
    <td class="tg-s4h7">8</td>
    <td class="tg-0udb">86.12</td>
    <td class="tg-s4h7">93.11</td>
    <td class="tg-s4h7">95.97</td>
    <td class="tg-s4h7">82.96</td>
    <td class="tg-s4h7">73.16</td>
    <td class="tg-0udb">88.16</td>
    <td class="tg-s4h7">93.59</td>
    <td class="tg-s4h7">95.73</td>
    <td class="tg-s4h7">82.53</td>
    <td class="tg-s4h7">70.20</td>
  </tr>
  <tr>
    <td class="tg-s4h7">9</td>
    <td class="tg-0udb">86.31</td>
    <td class="tg-s4h7">93.01</td>
    <td class="tg-s4h7">95.39</td>
    <td class="tg-s4h7">81.82</td>
    <td class="tg-s4h7">70.72</td>
    <td class="tg-0udb">84.81</td>
    <td class="tg-s4h7">93.16</td>
    <td class="tg-s4h7">95.53</td>
    <td class="tg-s4h7">80.04</td>
    <td class="tg-s4h7">67.35</td>
  </tr>
  <tr>
    <td class="tg-s4h7">10</td>
    <td class="tg-0udb">88.54</td>
    <td class="tg-s4h7">95.00</td>
    <td class="tg-s4h7">96.94</td>
    <td class="tg-s4h7">84.09</td>
    <td class="tg-s4h7">74.23</td>
    <td class="tg-0udb">87.14</td>
    <td class="tg-s4h7">93.93</td>
    <td class="tg-s4h7">96.21</td>
    <td class="tg-s4h7">83.04</td>
    <td class="tg-s4h7">70.77</td>
  </tr>
  <tr>
    <td class="tg-s4h7">Average</td>
    <td class="tg-0udb">86.43</td>
    <td class="tg-s4h7">94.10</td>
    <td class="tg-s4h7">96.25</td>
    <td class="tg-s4h7">82.51</td>
    <td class="tg-s4h7">72.33</td>
    <td class="tg-0udb">86.21</td>
    <td class="tg-s4h7">94.05</td>
    <td class="tg-s4h7">96.11</td>
    <td class="tg-s4h7">81.23</td>
    <td class="tg-s4h7">68.71</td>
  </tr>
</tbody>
</table>
