# Uncertainty-guided Label Correction with Wavelet-transformed Discriminative Representation Enhancement
This repository accompanies the research paper, [Uncertainty-guided Label Correction with Wavelet-transformed Discriminative Representation Enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003071) (accepted at Neural Networks 2024).
Our code is adapted from "[Extended T: Learning with Mixed Closed-set and Open-set Noisy Labels](https://ieeexplore.ieee.org/abstract/document/9790332)" (Zhang et al., IEEE Transactions on Pattern Analysis and Machine Intelligence 2022)

## ULTRA
Identifying noise is challenging because noisy samples closely resemble true positives. Existing approaches often assume a single noise source, oversimplify closed-set noise, or treat open-set noise as toxic and eliminate it, resulting in limited practical effects. To address these issues, we present a novel approach named uncertaintyguided label correction with wavelet-transformed discriminative representation enhancement (Ultra),
designed to mitigate the effects of mixed noise. To achieve robust mixed-noise identification, we initially look into a learnable wavelet filter for obtaining discriminative features and filtering spurious cues automatically at the representation level. Subsequently, we introduce a two-fold uncertainty estimation to stably locate noise within the corrupted supervised signal level. These insights pave the way for a simple yet potent label correction technique, enabling comprehensive utilization of open-set noise, which can be rendered non-toxic in a specific manner, in contrast to harmful closed-set noise. Experimental validation on datasets with synthetic mixed noise, web noise corruption, and a real-world dataset confirms the effectiveness and generality of Ultra. Furthermore, our approach enhances the application of efficient techniques (e.g., supervised contrastive learning) within label noise scenarios.

## Environments
You can setup python environment with:
```
conda create -n ULTRA python=3.6.15
conda activate ULTRA
pip install -r requirements.txt
```

## Data
Our experiment was conducted on the following datasets:
+ CIFAR-10
+ Red Mini-ImageNet
+ NoisywikiHow
+ Clothing1M

Opening images is time-consuming, so we need to first convert them to numpy format. You should get dataset as follows:
```
# Download data.zip from release, which contains data for `CIFAR-10` and `Red Mini-ImageNet`
cd ULTRA
mv /path/to/data.zip .
unzip data.zip

# get NoisywikiHow-dataset
git clone git@github.com:tangminji/NoisywikiHow-dataset.git
mv NoisywikiHow-dataset data/wikihow

# Follow the `step 1` from [this repo](https://github.com/Cysu/noisy_label) to get the Clothing1M data.
# get Clothing1M data
mkdir -p dataset
mv /path/to/clothing1M dataset/Clothing_1M
# move the image folder directly into Clothing_1M folder
mv dataset/Clothing_1M/images/* dataset/Clothing_1M/

# preprocess Clothing_1M dataset
conda activate ULTRA
python data_preprocess.py
```

## Running

To conduct our experiment, you need to run shell as follows:
```
#!/bin/bash

#SBATCH -J wiki_ours_0.4_nrun1-top5
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 20:00:00
#SBATCH -o results/wiki_ours_0.4_nrun1-top5.out

conda activate ULTRA

for seed in 0
do
    python main_ce1.py \
    --params_path enum/wiki/ours/0.4/param1/hy_best_params.json \  # best_params path, you can also set the params in command line according to the Table 7 in paper
    --dataset wiki \       # Available dataset: [cifar10s(default), cnwl(Red Mini-ImageNet), wiki(NoisywikiHow), Clothing1M]
    --model_type ours \    # Available method: ce(CrossEntropy, baseline), ours(ULTRA), ours_cl(ULTRA+)
    --noise_rate1 0.0 \    # open corruption rate
    --noise_rate2 0.4 \    # closed corruption rate
    --filter dwt \         # Available filter: None(for baseline), dwt(For ULTRA)
    --seed $seed \
    --f_type enh_red \
    --lam 0.5 \            # weight for representation enhancement
    --warm_up 6 \          # After warm_up epoch, Ultra starts to update 
    --epsilon 0.3 \        # for ID noise judgement
    --eta 0.3 \            # for OOD noise judgement
    --delta 0.006 \        # smoothing for one-hot vector
    --inc 0.0006 \         # for increment of epsilon, delta
    --n_epoch 20 \
    --nrun \               # set nrun, the results will save at 'nrun/dataset/xxx'
    --suffix top5          # extra info add to the path
done
```

Our best params can be found at Table 7 in our paper.

## Citation
If you find this code useful in your research then please cite:
```
@article{WU2024106383,
title = {Uncertainty-guided label correction with wavelet-transformed discriminative representation enhancement},
journal = {Neural Networks},
pages = {106383},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106383},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024003071},
author = {Tingting Wu and Xiao Ding and Hao Zhang and Minji Tang and Bing Qin and Ting Liu}}
```