import os
import re
import gc
import sys
import json
import random
import string
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForMaskedLM,
    get_linear_schedule_with_warmup
)


# global params
MODEL_CACHE_DIR = "./models"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


EFF_TYPE2ID = {
    "effective": 2,
    "adequate": 1,
    "ineffective": 0
}


def preprocess(args):
    # load contraction mapping json file
    with open(args.contraction_map, "r", encoding="utf-8") as f:
        contraction_map = json.load(f)
    punctuation = list(string.punctuation)
    # raw text normalizer
    def normalizer(s):
        def remove_punc(text):
            chs = []
            for ch in text:
                if ch not in punctuation:
                    chs.append(ch)
                else:
                    chs.append(" ")
            return "".join(chs)

        def lower(text):
            return text.lower()

        def replace_contractions(text):
            for k, v in contraction_map.items():
                text = text.replace(k, v)
                k = re.sub(r'\'', " '", k)
                text = text.replace(k, v)
            return text

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        _s = replace_contractions(s)
        _s = lower(_s)
        _s = remove_punc(_s)
        _s = remove_articles(_s)
        return white_space_fix(_s).strip()

    # read and clean raw train data
    if args.preprocess or not os.path.exists(args.trainset):
        train_raw_data = pd.read_csv(args.rawtrainset, sep=',')
        logger.info(f"Raw TRAIN csv file content:\n{train_raw_data.head(5)}")
        train_raw_data = train_raw_data.drop_duplicates(["discourse_text", "discourse_type", "discourse_effectiveness"])
        logger.info(f"length of valid TRAIN data: {len(train_raw_data)}", )
        train_raw_data["norm_discourse_text"] = train_raw_data["discourse_text"].map(normalizer)
        train_raw_data["norm_discourse_type"] = train_raw_data["discourse_type"].map(normalizer)
        train_raw_data["norm_discourse_effectiveness"] = train_raw_data["discourse_effectiveness"].map(lambda x: x.strip().lower()).map(lambda x: EFF_TYPE2ID[x])
        logger.info(f"Normalized TRAIN csv file content:\n{train_raw_data.head(5)}")
        logger.info("TRAIN dataset statistics:")
        logger.info(f"\n{train_raw_data.value_counts(['discourse_effectiveness'])}")
        logger.info(f"\n{train_raw_data.value_counts(['discourse_effectiveness', 'discourse_type'], sort=False)}")
        # save preprocessed data
        train_raw_data.to_csv(args.trainset, index=False)
    else:
        train_raw_data = pd.read_csv(args.trainset, sep=',')

    # read and clean raw test data
    if args.preprocess or not os.path.exists(args.testset):
        test_raw_data = pd.read_csv(args.rawtestset, sep=',')
        logger.info(f"Raw TEST csv file content:\n {test_raw_data.head(5)}")
        test_raw_data = test_raw_data.drop_duplicates(["discourse_text", "discourse_type"])
        logger.info("length of valid TEST data: %d", len(test_raw_data))
        test_raw_data["norm_discourse_text"] = test_raw_data["discourse_text"].map(normalizer)
        test_raw_data["norm_discourse_type"] = test_raw_data["discourse_type"].map(normalizer)
        logger.info(f"Normalized TEST csv file content:\n{test_raw_data.head(5)}")
        # save preprocessed data
        test_raw_data.to_csv(args.testset, index=False)
    else:
        test_raw_data = pd.read_csv(args.testset, sep=',')

    return train_raw_data, test_raw_data


class ArticleDataset(Dataset):

    def __init__(self, args, data: pd.DataFrame, label: pd.DataFrame=None):
        super(ArticleDataset, self).__init__()
        self.args = args
        # X: [N]
        self.X_ids = data["discourse_id"].values.tolist()
        self.X_essay_ids = data["essay_id"].values.tolist()
        self.X_rawtext = data["norm_discourse_text"].values.tolist()
        self.X_type = data["norm_discourse_type"]
        self.X_content = data["norm_discourse_text"]
        # Y: [N] numerical array
        self.is_train = label is not None
        if label is not None:
            self.Y = label["norm_discourse_effectiveness"].values.tolist()

    def __len__(self):
        return len(self.X_content)

    def __getitem__(self, i):
        if self.is_train:
            return self.X_type.iloc[i], self.X_content.iloc[i], self.Y[i]
        else:
            return self.X_type.iloc[i], self.X_content.iloc[i], 0


CLS_PREFIX = "the following discourse is a "
CLS_POSTFIX = " ."

PROMPT_PREFIX = "the following discourse ( "
PROMPT_POSTFIX = " ) is [MASK] ."


def cls_collator(batch_data):
    batch_seqs = []
    labels = []
    for data in batch_data:
        batch_seqs.append((CLS_PREFIX + data[0] + CLS_POSTFIX, data[1]))
        labels.append(data[2])
    return batch_seqs, torch.LongTensor(labels)


def prompt_collator(batch_data):
    batch_seqs = []
    labels = []
    for data in batch_data:
        batch_seqs.append((PROMPT_PREFIX + data[0] + PROMPT_POSTFIX, data[1]))
        labels.append(data[2])
    return batch_seqs, torch.LongTensor(labels)



class PretrainedClassificationModel(nn.Module):
    """
    Baselines: We use `[CLS]` encoding to pass through a [dropout -> dense(768xnum_labels)]
    """
    def __init__(self, args, device: torch.device):
        super(PretrainedClassificationModel, self).__init__()
        self.args = args
        self.device = device
        self.model_config = AutoConfig.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        logger.info(f"Model Backbone Config: \n{self.model_config}")
        logger.info("\nInitializing Pretraind Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        logger.info(f"tokenizer: {self.tokenizer}")
        logger.info("\nInitializing Pretraind Model...")
        self.backbone = AutoModel.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        # save vocab
        with open(os.path.join(MODEL_CACHE_DIR, "models--" + args.backbone.replace("/", "--"), "vocab.json"), "w+", encoding="utf-8") as f:
            json.dump(self.tokenizer.get_vocab(), f, indent=2)
        if args.freeze_backbone:
            self.classifier = nn.Sequential(
                nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
                nn.Tanh(),
                nn.Dropout(p=args.fc_dropout),
                nn.Linear(self.backbone.config.hidden_size, 3),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=args.fc_dropout),
                nn.Linear(self.backbone.config.hidden_size, 3),
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_sentences, labels=None):
        # for tokenizer options, see 
        # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertTokenizer
        # https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
        # convert texts to ids
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences,
            add_special_tokens=True,
            max_length=self.args.maxlen,
            truncation="only_second",
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=self.device)
        token_type_ids = torch.tensor(batch_tokenized["token_type_ids"], device=self.device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # model pipeline
        if self.args.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        encodings = outputs.last_hidden_state[:, 0] # use [CLS] embedding to classify, [B, C]
        logits = self.classifier(encodings)
        probs = F.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return probs, loss



class PromptPretrainedModel(nn.Module):

    def __init__(self, args, device: torch.device):
        super(PromptPretrainedModel, self).__init__()
        self.args = args
        self.device = device
        self.model_config = AutoConfig.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        logger.info(f"Model Backbone Config: \n{self.model_config}")
        logger.info("\nInitializing Pretraind Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        logger.info(f"tokenizer: {self.tokenizer}")
        logger.info("\nInitializing Pretraind Model...")
        self.backbone = AutoModelForMaskedLM.from_pretrained(args.backbone, cache_dir=MODEL_CACHE_DIR)
        # save vocab
        with open(os.path.join(MODEL_CACHE_DIR, "models--" + args.backbone.replace("/", "--"), "vocab.json"), "w+", encoding="utf-8") as f:
            json.dump(self.tokenizer.get_vocab(), f, indent=2)
        choice_map = {
            "effective": "effective",
            "adequate": "adequate",
            "ineffective": "ineffective",
            "valid": "effective",
            "true": "effective",
            "weak": "ineffective",
            "strong": "adequate",
            "sound": "effective",
        }
        choice_ids = []
        choice_id_map = []
        for choice_str, choice_str_mapped in choice_map.items():
            choice_id = self.tokenizer(choice_str).input_ids[1]
            choice_ids.append(choice_id)
            choice_id_map.append(EFF_TYPE2ID[choice_str_mapped])
        self.choice_ids = torch.tensor(choice_ids, device=device)
        self.choice_id_map = torch.tensor(choice_id_map, device=device)

    def forward(self, batch_sentences, labels=None):
        # for tokenizer options, see 
        # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertTokenizer
        # https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
        # convert texts to ids
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences,
            add_special_tokens=True,
            max_length=self.args.maxlen,
            truncation="only_second",
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=self.device)
        token_type_ids = torch.tensor(batch_tokenized["token_type_ids"], device=self.device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=self.device)
        if labels is not None:
            labels = labels.to(self.device)
        logits = self.backbone(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits # [B, L, E]
        masked_index = (input_ids == self.tokenizer.mask_token_id).nonzero()
        choice_logits = []
        for idx in masked_index:
            choice_logits.append(logits[idx[0], idx[1], self.choice_ids])
        choice_logits = torch.stack(choice_logits)
        probs = F.softmax(choice_logits, dim=-1) # B, num_choices
        # map logits into 3 classes
        squeezed_probs = torch.zeros(size=(probs.shape[0], 3), device=self.device)
        for i, choice_id in enumerate(self.choice_id_map):
            squeezed_probs[:, choice_id] += probs[:, i]
        squeezed_log_probs = squeezed_probs.log()
        if labels is not None:
            loss = F.nll_loss(squeezed_log_probs, labels)
        else:
            loss = None
        return squeezed_probs, loss



class TextCNN(nn.Module):

    def __init__(self, args, device: torch.device):
        super(TextCNN, self).__init__()
        self.device = device
        self.args = args

        embed_dim = 256
        class_num = 3
        kernel_num = 16
        kernel_sizes = [3, 4, 5]
        dropout = 0.5
        
        Ci = 1
        Co = kernel_num

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR)
        logger.info(f"tokenizer: {self.tokenizer}")
        self.embed = nn.Embedding(self.tokenizer.vocab_size, embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding = (2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_sentences, labels=None):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences,
            add_special_tokens=True,
            max_length=self.args.maxlen,
            truncation="only_second",
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # forward
        x = self.embed(input_ids)  # (N, token_num, embed_dim)
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        probs = F.softmax(logit, dim=-1)
        if labels is not None:
            loss = self.loss_fn(logit, labels)
        else:
            loss = None
        return probs, loss



class TextRNN(nn.Module):

    def __init__(self, args, device: torch.device):
        super(TextRNN, self).__init__()
        embed_dim = 256
        hidden_dim = 256
        class_num = 3
        bidirectional = True
        self.device = device
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR)
        logger.info(f"tokenizer: {self.tokenizer}")
        self.embeding = nn.Embedding(self.tokenizer.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=bidirectional, batch_first=True)
        self.l1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.ReLU()
        if bidirectional:
            self.l3 = nn.Linear(hidden_dim*2, class_num)
        else:
            self.l3 = nn.Linear(hidden_dim, class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, batch_sentences, labels=None):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences,
            add_special_tokens=True,
            max_length=self.args.maxlen,
            truncation="only_second",
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # forward
        x = self.embeding(input_ids)
        out, (final_hidden_state, _) = self.lstm(x)
        attn_output, _ = self.attention_net(out, final_hidden_state)
        logit = self.l3(attn_output)
        probs = F.softmax(logit, dim=-1)
        if labels is not None:
            loss = self.loss_fn(logit, labels)
        else:
            loss = None
        return probs, loss



def train_epoch(args, model, optimizer, scheduler, train_dataloader):
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Training...")
    total_train_loss_value = 0.0
    train_loss_value = 0.0
    tqdm_postfix = { "loss": np.nan, "bert_lr": np.nan, "cls_lr": np.nan, "norm": np.nan }
    pbar = tqdm(enumerate(train_dataloader, start=1), desc="train", total=len(train_dataloader), postfix=tqdm_postfix)
    model.train()
    for i, (seqs, labels) in pbar:
        _, loss = model(seqs, labels)
        total_train_loss_value += loss.item()
        loss = loss / args.gradient_accumulation_steps
        train_loss_value += loss.item()
        loss.backward()
        if i % args.gradient_accumulation_steps == 0:
            norm = clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tqdm_postfix["loss"] = train_loss_value
            tqdm_postfix["bert_lr"] = optimizer.state_dict()['param_groups'][0]['lr']
            tqdm_postfix["cls_lr"] = optimizer.state_dict()['param_groups'][-1]['lr']
            tqdm_postfix["norm"] = norm
            pbar.set_postfix(tqdm_postfix)
            train_loss_value = 0.0
    return total_train_loss_value / len(train_dataloader)



@torch.no_grad()
def valid_epoch(model, valid_dataloader):
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Validating...")
    valid_loss_value = 0.0
    acc_score = 0.0
    recall_score = 0.0
    f1_score = 0.0
    model.eval()
    for i, (seqs, labels) in tqdm(enumerate(valid_dataloader, start=1), desc="valid", total=len(valid_dataloader)):
        p, loss = model(seqs, labels)
        valid_loss_value += loss.item()
        labels = labels.tolist()
        predictions = torch.argmax(p, dim=-1).cpu().tolist()
        acc_score += metrics.precision_score(y_true=labels, y_pred=predictions, labels=[0, 1, 2], average="micro")
        recall_score += metrics.recall_score(y_true=labels, y_pred=predictions, labels=[0, 1, 2], average="macro")
        f1_score += metrics.f1_score(y_true=labels, y_pred=predictions, labels=[0, 1, 2], average="macro")
    return valid_loss_value / len(valid_dataloader), acc_score / len(valid_dataloader), recall_score / len(valid_dataloader), f1_score / len(valid_dataloader)



@torch.no_grad()
def test(model, test_dataloader):
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Testing...")
    model.eval()
    probs = []
    for i, (seqs, _) in tqdm(enumerate(test_dataloader, start=1), desc="test", total=len(test_dataloader)):
        p, _ = model(seqs)
        probs.append(p)
    return torch.cat(probs, dim=0).cpu().numpy() # [N, 3]



def save_predictions(dataset, predictions, output_file):
    assert len(dataset) == predictions.shape[0]
    data = {
        "discourse_id": [],
        "Ineffective": [],
        "Adequate": [],
        "Effective": [],
    }
    for i, discourse_id in enumerate(dataset.X_ids):
        data["discourse_id"].append(discourse_id)
        data["Ineffective"].append(predictions[i][0])
        data["Adequate"].append(predictions[i][1])
        data["Effective"].append(predictions[i][2])
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def save_model(path, epoch, model, optimizer, scheduler):
    state_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state_dict, path)


def set_optimization_params(model):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.backbone_lr,
        },
        {
            "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0 if args.no_ln_decay else args.weight_decay,
            "lr": args.backbone_lr,
        },
        {
            "params": filter(lambda p: id(p) not in list(map(id, model.backbone.parameters())), model.parameters()),
            "weight_decay": args.weight_decay,
            "lr": args.classifier_lr,
        },
    ]
    return optimizer_grouped_parameters


MODEL_MAP = {
    "bert-base-uncased": PretrainedClassificationModel, # https://huggingface.co/bert-base-uncased
    "microsoft/deberta-v3-large": PretrainedClassificationModel, # https://huggingface.co/microsoft/deberta-v3-large
    "microsoft/deberta-v3-base": PretrainedClassificationModel, # https://huggingface.co/microsoft/deberta-v3-base
    "microsoft/deberta-v3-small": PretrainedClassificationModel, # https://huggingface.co/microsoft/deberta-v3-small
    "microsoft/deberta-v3-xsmall": PretrainedClassificationModel, # https://huggingface.co/microsoft/deberta-v3-xsmall
    "textcnn": TextCNN,
    "textrnn": TextRNN,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Discourse Classification")
    parser.add_argument("--seed", type=int, default=23333333)
    # datasets
    parser.add_argument("--trainset", type=str, default="./data/train_cleaned.csv")
    parser.add_argument("--testset", type=str, default="./data/test_cleaned.csv")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--rawtrainset", type=str, default="./data/train.csv")
    parser.add_argument("--rawtestset", type=str, default="./data/test.csv")
    parser.add_argument("--contraction_map", type=str, default="./contraction_map.json")
    # checkpoints and log saving directory
    parser.add_argument("--basedir", type=str, default="./log")
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    # model
    parser.add_argument("--backbone", type=str, required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--fc_dropout", type=float, default=0.1)
    # fine-tuning options
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=2e-4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--no_ln_decay", action="store_true")
    parser.add_argument("--group_params", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_bsz", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--kfold", type=int, default=1)
    # test
    # checkpoint
    parser.add_argument("--checkpoint", type=str, default=None, action="append")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--result_csv", type=str, default="result.csv")

    args = parser.parse_args()
    return args


def init_logger(logdir):
    logger = logging.getLogger("default")
    cmd_handler = logging.StreamHandler(sys.stdout)
    cmd_handler.setLevel(logging.DEBUG)
    cmd_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    log_handler = logging.FileHandler(os.path.join(logdir, "train.log"), mode="w+", encoding="utf-8")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    logger.addHandler(cmd_handler)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make logging directory
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(args.basedir, exist_ok=True)
    logging_dir = os.path.join(args.basedir, args.expname)
    if os.path.exists(logging_dir) and not args.overwrite:
        print(f"[WARN] logging directory {logging_dir} already exists. If you want to overwrite previous logs, use param `--overwrite` please.")
        exit(-1)
    if args.prompt and (args.backbone == "textcnn" or args.backbone == "textrnn"):
        print(f"[WARN] Backbone {args.backbone} does not support prompting !")
        exit(-1)
    
    os.makedirs(logging_dir, exist_ok=True)
    # init logging module
    logger = init_logger(logging_dir)
    # save configs
    with open(os.path.join(logging_dir, "config.json"), "w+", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # dataset preprocessing
    logger.info(f"Loading datasets")
    train_raw_data, test_raw_data = preprocess(args)
    train_x = train_raw_data[["discourse_id", "essay_id", "norm_discourse_text", "norm_discourse_type"]]
    train_y = train_raw_data[["norm_discourse_effectiveness"]]

    # choose collator
    collator = prompt_collator if args.prompt else cls_collator

    # test dataset
    test_dataset = ArticleDataset(args, data=test_raw_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    logger.info(f"[DATASET] test  set size = {len(test_dataset )}")

    if not args.test_only:
        # train and validation and test
        # k-fold trainning
        real_batch_size = args.train_bsz * args.gradient_accumulation_steps
        logger.info(f"Batch Size={real_batch_size} (Accumulation Steps={args.gradient_accumulation_steps}), optimization iters={len(train_x)//real_batch_size}")
        if args.kfold > 1:
            skf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
            fold_accs = []
            fold_loss = []
            probs_each = []
            logger.info("Start Training Process")
            for n_fold, (train_idxs, valid_idxs) in enumerate(skf.split(train_x, train_y)):
                logger.info(f"[FOLD] {n_fold}-th fold...")
                sub_train_x, sub_train_y = train_x.iloc[train_idxs], train_y.iloc[train_idxs]
                sub_valid_x, sub_valid_y = train_x.iloc[valid_idxs], train_y.iloc[valid_idxs]
                train_dataset = ArticleDataset(args, data=sub_train_x, label=sub_train_y)
                valid_dataset = ArticleDataset(args, data=sub_valid_x, label=sub_valid_y)
                train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=collator)
                valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_bsz, shuffle=False, drop_last=False, num_workers=args.num_workers, collate_fn=collator)
                logger.info(f"[DATASET] train set size = {len(train_dataset)}")
                logger.info(f"[DATASET] valid set size = {len(valid_dataset)}")
                # model construction
                num_training_steps = args.epoch * len(train_dataset) // real_batch_size + 1
                if args.prompt:
                    model = PromptPretrainedModel(args, device).to(device)
                else:
                    model = MODEL_MAP[args.backbone](args, device).to(device)
                optimizer_grouped_parameters = set_optimization_params(model) if args.group_params else model.parameters()
                optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.backbone_lr, eps=args.eps, betas=(args.beta1, args.beta2))
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
                # trainning process
                for epoch in range(1, args.epoch+1):
                    logger.info(f"[EPOCH] {epoch}")
                    # train
                    train_avg_loss = train_epoch(args, model, optimizer, scheduler, train_dataloader)
                    logger.info(f"[TRAIN] loss={train_avg_loss}")
                    save_model(os.path.join(logging_dir, f"kfold_{n_fold}_epoch_{epoch}.pt"), epoch, model, optimizer, scheduler)
                    # validate
                    valid_avg_loss, valid_acc, valid_recall, valid_f1 = valid_epoch(model, valid_dataloader)
                    logger.info(f"Fold {n_fold} | Epoch {epoch}: valid_loss={valid_avg_loss}, valid acc={valid_acc}, valid recall={valid_recall}, valid f1={valid_f1}")
                fold_accs.append(valid_acc)
                fold_loss.append(valid_avg_loss)
                # output test prediction
                probs = test(model, test_dataloader)
                probs_each.append(probs)
                del model
            logger.info(f"{args.kfold}-fold cross validation loss={np.mean(fold_loss)}(std={np.std(fold_loss)}), cross validation acc = {np.mean(fold_accs)}(std={np.std(fold_accs)})")
            # k-fold model ensemble (mean method)
            probs_each = np.array(probs_each) # [K, N, 3]
            final_probs = np.mean(probs_each, axis=0) # [N, 3]
            logger.info(f"final probs: {final_probs}")
            save_predictions(test_dataset, final_probs, os.path.join(logging_dir, args.result_csv))
        else:
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
            train_dataset = ArticleDataset(args, data=train_x, label=train_y)
            valid_dataset = ArticleDataset(args, data=valid_x, label=valid_y)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
            logger.info(f"[DATASET] train set size = {len(train_dataset)}")
            logger.info(f"[DATASET] valid set size = {len(valid_dataset)}")
            # training settings
            num_training_steps = args.epoch * len(train_dataset) // real_batch_size + 1
            if args.prompt:
                model = PromptPretrainedModel(args, device).to(device)
            else:
                model = MODEL_MAP[args.backbone](args, device).to(device)
            optimizer_grouped_parameters = set_optimization_params(model) if args.group_params else model.parameters()
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.backbone_lr, eps=args.eps, betas=(args.beta1, args.beta2))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
            logger.info("Start Training Process")
            for epoch in range(1, args.epoch+1):
                logger.info(f"[EPOCH] {epoch}")
                # train
                train_avg_loss = train_epoch(args, model, optimizer, scheduler, train_dataloader)
                logger.info(f"[TRAIN] loss={train_avg_loss}")
                save_model(os.path.join(logging_dir, f"no_fold_epoch_{epoch}.pt"), epoch, model, optimizer, scheduler)
                # validate
                valid_avg_loss, valid_acc, valid_recall, valid_f1 = valid_epoch(model, valid_dataloader)
                logger.info(f"No Fold | Epoch {epoch}: valid_loss={valid_avg_loss}, valid acc={valid_acc}, valid recall={valid_recall}, valid f1={valid_f1}")
            probs = test(model, test_dataloader) # [N, 3]
            logger.info(f"final_predictions: {probs}")
            save_predictions(test_dataset, probs, os.path.join(logging_dir, args.result_csv))
    else:
        gc.collect()
        torch.cuda.empty_cache()
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
        train_dataset = ArticleDataset(args, data=train_x, label=train_y)
        valid_dataset = ArticleDataset(args, data=valid_x, label=valid_y)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        logger.info(f"[DATASET] train set size = {len(train_dataset)}")
        logger.info(f"[DATASET] valid set size = {len(valid_dataset)}")
        
        # only inference
        if not args.checkpoint:
            logger.info(f"[MODEL] Load params from pre-trained model: {args.backbone}")
            if args.prompt:
                model = PromptPretrainedModel(args, device).to(device)
            else:
                model = MODEL_MAP[args.backbone](args, device).to(device)
            valid_avg_loss, valid_acc, valid_recall, valid_f1 = valid_epoch(model, valid_dataloader)
            logger.info(f"valid_loss={valid_avg_loss}, valid acc={valid_acc}, valid recall={valid_recall}, valid f1={valid_f1}")
            probs = test(model, test_dataloader)
            save_predictions(test_dataset, probs, os.path.join(logging_dir, args.result_csv))
        elif len(args.checkpoint) == 1:
            state_dict = torch.load(args.checkpoint[0])
            logger.info(f"[MODEL] Load params from checkpoint: {args.checkpoint[0]}")
            if args.prompt:
                model = PromptPretrainedModel(args, device).to(device)
            else:
                model = MODEL_MAP[args.backbone](args, device).to(device)
            model.load_state_dict(state_dict["model"])
            valid_avg_loss, valid_acc, valid_recall, valid_f1 = valid_epoch(model, valid_dataloader)
            logger.info(f"valid_loss={valid_avg_loss}, valid acc={valid_acc}, valid recall={valid_recall}, valid f1={valid_f1}")
            probs = test(model, test_dataloader)
            save_predictions(test_dataset, probs, os.path.join(logging_dir, args.result_csv))
        elif len(args.checkpoint) > 1:
            # ensemble models
            probs_each = []
            for ckpt in args.checkpoint:
                state_dict = torch.load(ckpt)
                logger.info(f"[MODEL] Load params from checkpoint: {ckpt}")
                gc.collect()
                torch.cuda.empty_cache()
                if args.prompt:
                    model = PromptPretrainedModel(args, device).to(device)
                else:
                    model = MODEL_MAP[args.backbone](args, device).to(device)
                model.load_state_dict(state_dict["model"])
                valid_avg_loss, valid_acc, valid_recall, valid_f1 = valid_epoch(model, valid_dataloader)
                logger.info(f"valid_loss={valid_avg_loss}, valid acc={valid_acc}, valid recall={valid_recall}, valid f1={valid_f1}")
                probs = test(model, test_dataloader)
                probs_each.append(probs)
                del model
            # models ensemble (mean method)
            probs_each = np.array(probs_each) # [K, N, 3]
            final_probs = np.mean(probs_each, axis=0) # [N, 3]
            logger.info(f"final probs: {final_probs}")
            save_predictions(test_dataset, final_probs, os.path.join(logging_dir, args.result_csv))
        else:
            raise ValueError(f"`args.checkpoint` should be non-empty when `args.test_only == True`.")

