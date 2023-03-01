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

MAX_SOURCE_LENGTH=512

def pad_assert(tokenizer, source_ids):
    source_ids = source_ids[:MAX_SOURCE_LENGTH - 2]
    source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    pad_len = MAX_SOURCE_LENGTH - len(source_ids)
    source_ids += [tokenizer.pad_id] * pad_len
    assert len(source_ids) == MAX_SOURCE_LENGTH, "Not equal length."
    return source_ids

def encode_diff(tokenizer, diff):
    difflines = diff.split("\n")[1:]        # remove start @@
    difflines = [line for line in difflines if len(line.strip()) > 0]
    map_dic = {"-": 0, "+": 1, " ": 2}
    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2
    labels = [f(line[0]) for line in difflines]
    difflines = [line[1:].strip() for line in difflines]
    inputstr = ""
    for label, line in zip(labels, difflines):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<del>" + line
        else:
            inputstr += "<keep>" + line
    source_ids = tokenizer.encode(inputstr, max_length=MAX_SOURCE_LENGTH, truncation=True)[1:-1]
    source_ids = pad_assert(tokenizer, source_ids)
    return source_ids

parser = argparse.ArgumentParser()
args = add_args(parser)
set_seed(args)
config, model, tokenizer = build_or_load_gen_model(args)
model_size = get_model_size(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model_size)
print("Model device:", model.device)
model.eval()
code_diff = """@@ -15,7 +15,7 @@ public class ManipulationTest extends BasicJBehaveTest {\n \n     @Override\n     public InjectableStepsFactory stepsFactory() {\n-        Map<String, Object> state = new HashMap<String, Object>();\n+        Map<String, Object> state = new HashMap<>();\n \n         return new InstanceStepsFactory(configuration(),\n                 new SharedSteps(state)"""
inputs = torch.tensor([encode_diff(tokenizer, code_diff)], dtype=torch.long).to("cuda")
inputs_mask = inputs.ne(tokenizer.pad_id)
preds = model.generate(inputs,
                        attention_mask=inputs_mask,
                        use_cache=True,
                        num_beams=5,
                        early_stopping=True,
                        max_length=100,
                        num_return_sequences=5
                        )
preds = list(preds.cpu().numpy())
pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in preds]
print(code_diff)
print(pred_nls[0])
print(pred_nls)