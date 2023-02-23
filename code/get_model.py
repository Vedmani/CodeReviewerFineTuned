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
from models import build_or_load_gen_model, get_model_size
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser()
args = add_args(parser)
set_seed(args)
config, model, tokenizer = build_or_load_gen_model(args)
model_size = get_model_size(model)
model.cuda()
print(model_size)
print("Model device:", model.device)