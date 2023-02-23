import os
import torch
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import f1_score, accuracy_score


"""logging.basicConfig(
    filename="example.log",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('example.log')
fh.setLevel(logging.DEBUG)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def main(args):
    """local_rank = 0
    args.global_rank = 0
    args.local_rank = local_rank
    args.world_size = 1"""
    # device = "cuda" if torch.cuda.is_availablee() else "cpu"
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    """if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)):
        model.load_state_dict(
            torch.load("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
        )"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    print("Optimizer loaded.")
    logger.debug("Optimizer loaded")
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
    global_step = 0
    save_steps = args.save_steps
    train_file = args.train_filename
    valid_file = args.dev_filename
    if os.path.isdir(train_file):
        train_files = [file for file in os.listdir(train_file) if
                       file.startswith("cls-train-chunk") and file.endswith(".jsonl")]
    else:
        train_files = [train_file]
    logger.warning("Train files: %s", train_files)
    random.seed(args.seed)
    random.shuffle(train_files)
    train_files = [os.path.join(train_file, file) for file in train_files]
    valid_files = [valid_file]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.device)
    logger.debug(args)
    main(args)
    logger.info("Training finished.")