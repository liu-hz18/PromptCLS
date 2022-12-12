# Improving BERTs' Classification Performance on Writing Arguments using Prompting
Hongzun Liu, Xuetong Wang, Jie Wu

### This work is a course project of Machine Learning (2022 Fall) at Tsinghua University

## Preparation
You may need a conda environment and CUDA on your machine. About 40GB GPU memory is required.
```
pip install -r requirements.txt
```

## Fine-tuning head-based models
You may just use default options.
```
python main.py --expname bert-nofold-exp1 --backbone bert-base-uncased --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
```

## Fine-tuning prompt-based models
You may use `--prompt` option.
```
python main.py --expname deberta-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
```

## K-fold validation
You may use `--kfold {K}` option.
```
python main.py --expname deberta-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 10
```

## Test only
You may use `--test_only` option.
```
python main.py --expname deberta-nofold-prompt-exp1-testonly --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1 --test_only
```

## Train a TextCNN / TextRNN baseline
TextRNN
```
python main.py --expname textrnn-nofold-exp1 --backbone textrnn --epoch 15 --train_bsz 64 --eval_bsz 128 --kfold 1 --backbone_lr 5e-4
```
TextCNN
```
python main.py --expname textcnn-nofold-exp1 --backbone textcnn --epoch 15 --train_bsz 64 --eval_bsz 128 --kfold 1 --backbone_lr 5e-4
```

## Reproduction
All of our experiments can be reproduced using scripts in `train_all.sh`
```
CUDA_VISIBLE_DEVICES="0" sh ./train_all.sh
```
