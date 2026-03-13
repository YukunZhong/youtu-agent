#!/usr/bin/env python3
"""Quick test of the Anole server with MODEL_TOKENS (image prefix only)."""
import json
import numpy as np
import requests

# Load first sample from ScienceQA train
with open("/data1/data/kangborui/zhongyukun/medmax/MoDE-official/data/ScienceQA/train_data.jsonl") as f:
    sample = json.loads(f.readline())
tokens = np.array(sample["tokens"], dtype=np.int64)

# Only send image prefix: BOS + BOI + 1024 img tokens + EOI = first 1027 tokens
image_prefix = tokens[:1027]
token_str = ",".join(str(int(t)) for t in image_prefix)
print(f"Image prefix length: {len(image_prefix)}")
print(f"Full tokens length: {len(tokens)}")
print(f"Question text: {sample['text'][:200]}")
print(f"Answer: {sample['answer']}")

# Extract question from text (remove <image> tag and reserved tokens)
import re
question = sample["text"].replace("<image>", "").strip()
question = re.sub(r"<reserved\d+>.*", "", question).strip()

# Build prompt: image tokens + question text
prompt = "Answer with the option's letter from the given choices directly."
msg_content = f"[MODEL_TOKENS]\n{token_str}\n[/MODEL_TOKENS]\n\n{question}\n{prompt}"

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anole-7b",
        "messages": [{"role": "user", "content": msg_content}],
        "temperature": 0.7,
        "max_tokens": 32,
    },
    proxies={"http": None, "https": None},
)
print(f"\nStatus: {resp.status_code}")
result = resp.json()
print(f"Response: {json.dumps(result, indent=2)}")
