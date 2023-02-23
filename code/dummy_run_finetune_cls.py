import logging
import argparse
import torch
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


def main(args):
    local_rank = 0
    args.global_rank = 0
    args.local_rank = local_rank
    args.world_size = 1
    device = "cuda" if torch.cuda.is_availablee() else "cpu"
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Training finished.")