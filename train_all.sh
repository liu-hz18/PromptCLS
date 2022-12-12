# textrnn
python main.py --expname textrnn-nofold-exp1 --backbone textrnn --epoch 15 --train_bsz 64 --eval_bsz 128 --kfold 1 --backbone_lr 5e-4
# textcnn
python main.py --expname textcnn-nofold-exp1 --backbone textcnn --epoch 15 --train_bsz 64 --eval_bsz 128 --kfold 1 --backbone_lr 5e-4
# bert-head
python main.py --expname bert-nofold-exp1 --backbone bert-base-uncased --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# bert-head-freeze
python main.py --expname bert-nofold-freeze-exp1 --backbone bert-base-uncased --epoch 8 --train_bsz 64 --eval_bsz 64 --kfold 1 --freeze_backbone
# deberta-head
python main.py --expname deberta-nofold-exp1 --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-head-freeze
python main.py --expname deberta-nofold-freeze-exp1 --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 64 --eval_bsz 64 --kfold 1 --freeze_backbone
# bert-prompt
python main.py --expname bert-nofold-prompt-exp1 --prompt --backbone bert-base-uncased --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-prompt
python main.py --expname deberta-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1

# fine-tuning methods
# deberta-zero-shot
python main.py --expname deberta-zeroshot-nofold-prompt-exp1 --prompt --test_only --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-few-shot
python main.py --expname deberta-fewshot-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 1 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-fine-tune -> deberta-prompt
# BLANK

# parameter update methods
# promptless -> deberta-prompt
# BLANK
# tuning-free -> deberta-zero-shot
# BLANK
# fixed-lm: p-tuning (prefix-length = 8)
python main.py --expname deberta-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# prompt+lm fine-tune: p-tuning (prefix-length = 8)
python main.py --expname deberta-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-base --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1

# model size
# deberta-xsmall
python main.py --expname deberta-xsmall-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-xsmall --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-small
python main.py --expname deberta-small-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-small --epoch 8 --train_bsz 32 --gradient_accumulation_steps 2 --eval_bsz 64 --kfold 1
# deberta-base -> deberta-prompt
# BLANK
# deberta-large
python main.py --expname deberta-large-nofold-prompt-exp1 --prompt --backbone microsoft/deberta-v3-large --epoch 8 --train_bsz 16 --gradient_accumulation_steps 4 --eval_bsz 32 --kfold 1
