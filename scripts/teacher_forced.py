"""Teacher-forced accuracy: feed the real prompt + completion, compare argmax next-token
predictions to the true motion tokens, per RVQ level. ~100 % means the LLM memorized its
training rows, so any gap to the tokenizer ceiling comes from free generation (sampling /
exposure bias), not from under-fitting.

usage: python scripts/teacher_forced.py <lora_checkpoint_dir> <train_jsonl> <num_quantizers> [n_random_rows=9]
Row 0 (BWW760) is always included.
"""
import os, sys, json, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from speech_to_motion_pipeline import add_discrete_tokens, encode_for_training

ckpt, jsonl, num_q = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 9

model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit", max_seq_length=8192, load_in_4bit=True)
tok = add_discrete_tokens(tok, motion_num_codebooks=num_q)
model.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(model, ckpt)
FastLanguageModel.for_inference(model)

rows = [json.loads(l) for l in open(jsonl)]
random.seed(0)
pick = [0] + random.sample(range(1, len(rows)), min(n, len(rows) - 1))
overall = []
with torch.no_grad():
    for i in pick:
        enc = encode_for_training(rows[i], tok, max_seq_length=8192)
        ids = torch.tensor([enc["input_ids"]], device="cuda")
        pred = model(input_ids=ids).logits[0, :-1].argmax(-1).cpu()
        true = torch.tensor(enc["input_ids"][1:])
        mask = torch.tensor(enc["labels"][1:]) != -100
        pos = torch.where(mask)[0][:-1]          # motion tokens only, drop the trailing EOS
        per_level = [(pred[pos[q::num_q]] == true[pos[q::num_q]]).float().mean().item() for q in range(num_q)]
        acc = (pred[pos] == true[pos]).float().mean().item()
        overall.append(acc)
        print(f"row {i:4d}: acc {acc:.3f}  per-level {[round(v, 3) for v in per_level]}  ({len(pos)} tok)", flush=True)
print(f"TEACHER_FORCED row0={overall[0]:.3f} mean={sum(overall) / len(overall):.3f} rows={len(overall)}")
