import argparse
import torch
from configs import add_args
from models import ReviewerModel, build_or_load_gen_model

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
args.model_name_or_path = "microsoft/codereviewer"
config, model, tokenizer = build_or_load_gen_model(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
code_diff = """@@ -18,8 +18,11 @@ from mitmproxy import io\n from mitmproxy import log\n from mitmproxy import version\n from mitmproxy import optmanager\n+from mitmproxy import options\n import mitmproxy.tools.web.master # noqa\n \n+CONFIG_PATH = os.path.join(options.CA_DIR, \'config.yaml\')\n+\n \n def flow_to_json(flow: mitmproxy.flow.Flow) -> dict:\n"""

inputs = torch.tensor([encode_diff(tokenizer, code_diff)], dtype=torch.long).to(device)
inputs_mask = inputs.ne(tokenizer.pad_id)
preds = model.generate(inputs,
                        attention_mask=inputs_mask,
                        use_cache=True,
                        num_beams=5,
                        early_stopping=True,
                        max_length=100,
                        num_return_sequences=4
                        )
preds = list(preds.cpu().numpy())
pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in preds]
print(code_diff)
print(pred_nls[0])
print(pred_nls)
    