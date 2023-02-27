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
code_diff = """@@ -34,24 +34,34 @@ func (host *HostV2) Peerstore() peerstore.Peerstore {\n }\n \n // New creates a host for p2p communication\n-func New(self p2p.Peer) *HostV2 {\n-\tsourceAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%s", self.Port))\n-\tcatchError(err)\n+func New(self p2p.Peer, priKey p2p_crypto.PrivKey) *HostV2 {\n+\n+\t// TODO (leo), use the [0] of Addrs for now, need to find a reliable way of using listenAddr\n \tp2pHost, err := libp2p.New(context.Background(),\n-\t\tlibp2p.ListenAddrs(sourceAddr),\n-\t\tlibp2p.NoSecurity, // The security (signature generation and verification) is, for now, taken care by ourselves.\n+\t\tlibp2p.ListenAddrs(self.Addrs[0]),\n+\t\tlibp2p.Identity(priKey),\n \t\t// TODO(ricl): Other features to probe\n \t\t// libp2p.EnableRelay; libp2p.Routing;\n \t)\n+\n \tcatchError(err)\n-\tlog.Debug("HostV2 is up!", "port", self.Port, "id", p2pHost.ID().Pretty(), "addr", sourceAddr)\n+\tlog.Debug("HostV2 is up!", "port", self.Port, "id", p2pHost.ID().Pretty(), "addr", self.Addrs)\n+\n+\t// has to save the private key for host\n \th := &HostV2{\n-\t\th:    p2pHost,\n-\t\tself: self,\n+\t\th:      p2pHost,\n+\t\tself:   self,\n+\t\tpriKey: priKey,\n \t}\n+\n \treturn h\n }\n \n+// GetID returns ID.Pretty\n+func (host *HostV2) GetID() peer.ID {\n+\treturn host.h.ID()\n+}\n+\n // GetSelfPeer gets self peer\n func (host *HostV2) GetSelfPeer() p2p.Peer {\n \treturn host.self"""
inputs = torch.tensor([encode_diff(tokenizer, code_diff)], dtype=torch.long).to("cuda")
inputs_mask = inputs.ne(tokenizer.pad_id)
preds = model.generate(inputs,
                        attention_mask=inputs_mask,
                        use_cache=True,
                        num_beams=5,
                        early_stopping=True,
                        max_length=100,
                        num_return_sequences=2
                        )
preds = list(preds.cpu().numpy())
pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in preds]
print(pred_nls[0])