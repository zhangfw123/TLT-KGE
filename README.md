## TLT-KGE CIKM 2022
This repo is for source code of CIKM 2022 paper "Along the Time: Timeline-traced Embedding for Temporal Knowledge Graph Completion".
Paper Link: https://dl.acm.org/doi/pdf/10.1145/3511808.3557233 (CIKM 2022).

## Environment

- tqdm==4.59.0
- numpy==1.20.1
- scikit-learn==0.24.1
- scipy==1.6.2
- torch==1.9.0


## Download Datasets

You can download the datasets from https://drive.google.com/file/d/1uzyA1liRRqrrrq3oeL-ZIVOr1ot1k0et/view?usp=sharing. 

## Before Experiments

Install the kbc package.
```
python setup.py install
```

Preprocess the datasets.
```
python tkbc/process_icews.py
python tkbc/process_gdelt.py 
```

## Run the Experiments

```
python tkbc/learner.py --dataset ICEWS14 --model TLT_KGE_Quaternion --rank 1200 --emb_reg 3e-3 --time_reg 3e-2 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --gpu 1
python tkbc/learner.py --dataset ICEWS14 --model TLT_KGE_Complex --rank 1200 --emb_reg 1e-3 --time_reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --gpu 1
python tkbc/learner.py --dataset ICEWS05-15 --model TLT_KGE_Quaternion --rank 1200 --emb_reg 1e-3 --time_reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 1440 --gpu 1
python tkbc/learner.py --dataset ICEWS05-15 --model TLT_KGE_Complex --rank 1200 --emb_reg 1e-3 --time_reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 1440 --gpu 1
python tkbc/learner.py --dataset gdelt --model TLT_KGE_Quaternion --rank 1500 --emb_reg 5e-4 --time_reg 3e-2 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --gpu 1
python tkbc/learner.py --dataset gdelt --model TLT_KGE_Complex --rank 1500 --emb_reg 1e-3 --time_reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --gpu 1
```

## Citation
```
@inproceedings{zhang2022along,
  title={Along the Time: Timeline-traced Embedding for Temporal Knowledge Graph Completion},
  author={Zhang, Fuwei and Zhang, Zhao and Ao, Xiang and Zhuang, Fuzhen and Xu, Yongjun and He, Qing},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={2529--2538},
  year={2022}
}
```


## Acknowledge
TLT-KGE is based on tkbc: https://github.com/facebookresearch/tkbc.
