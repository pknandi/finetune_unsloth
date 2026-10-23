"""Audio-shuffle test: does the LLM use the audio? For held-out rows, the mean token loss (NLL) of
the true motion tokens given the row's own audio vs given another row's audio. A positive gap
(wrong - right) means the audio helps predict motion; ~0 means the model ignores it. Unlike
held-out MPJPE this doesn't punish a plausible-but-different gesture, and the gap (in nats per
token) can be compared across runs and data scales.

usage: python scripts/audio_shuffle.py <lora_checkpoint_dir> <val_jsonl> <num_quantizers> [n_rows=40]
Rows: n random rows (seed 0). Wrong audio for row k is the prompt of the picked row k + n//2 (mod n).
Last line: AUDIO_SHUFFLE right=<nll> wrong=<nll> gap=<wrong-right> gap_l0=<level-0 gap> win=<rows with gap>0>/<n>
"""
import os, sys, json, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from peft import PeftModel
from speech_to_motion_pipeline import add_discrete_tokens, encode_for_training

ckpt, jsonl, num_q = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 40

model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit", max_seq_length=8192, load_in_4bit=True)
tok = add_discrete_tokens(tok, motion_num_codebooks=num_q)
model.resize_token_embeddings(len(tok), mean_resizing=False)   # trained rows come from the checkpoint
model = PeftModel.from_pretrained(model, ckpt)
FastLanguageModel.for_inference(model)

rows = [json.loads(l) for l in open(jsonl)]
random.seed(0)
pick = random.sample(range(len(rows)), min(n, len(rows)))
n = len(pick)

def nll(prompt, completion):
    """Per-token NLL of the motion tokens (trailing EOS dropped), and the level-0 subset."""
    enc = encode_for_training({"prompt": prompt, "completion": completion}, tok, max_seq_length=8192)
    ids = torch.tensor([enc["input_ids"]], device="cuda")
    labels = torch.tensor(enc["labels"][1:], device="cuda")
    pos = torch.where(labels != -100)[0][:-1]
    logits = model(input_ids=ids).logits[0, :-1][pos].float()
    loss = F.cross_entropy(logits, labels[pos], reduction="none")
    return loss.mean().item(), loss[0::num_q].mean().item()

res = []
with torch.no_grad():
    for k, i in enumerate(pick):
        j = pick[(k + n // 2) % n]
        r, r0 = nll(rows[i]["prompt"], rows[i]["completion"])
        w, w0 = nll(rows[j]["prompt"], rows[i]["completion"])
        res.append((r, w, r0, w0))
        print(f"row {i:4d}: right {r:.3f}  wrong {w:.3f}  gap {w - r:+.3f}  (level 0: {r0:.3f} vs {w0:.3f})", flush=True)
R = sum(x[0] for x in res) / n; W = sum(x[1] for x in res) / n
R0 = sum(x[2] for x in res) / n; W0 = sum(x[3] for x in res) / n
print(f"AUDIO_SHUFFLE right={R:.3f} wrong={W:.3f} gap={W - R:+.3f} gap_l0={W0 - R0:+.3f} "
      f"win={sum(x[1] > x[0] for x in res)}/{n}")
