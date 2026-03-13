#!/usr/bin/env python3
"""
Multi-GPU parallel OpenAI-compatible API server for Anole.

Launches one model instance per GPU as a separate subprocess.
The main process runs a FastAPI load-balancer that distributes requests
round-robin across all GPU workers for true parallel inference.

Supports [MODEL_TOKENS] tags for passing pre-tokenized inputs (MoDE format).

Usage:
    # Use all 6 GPUs:
    python scripts/serve_anole.py --port 8000 --gpus 0,1,2,3,4,5

    # Use specific GPUs:
    python scripts/serve_anole.py --port 8000 --gpus 0,4,5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────

MODEL_PATH = "/data1/data/kangborui/zhongyukun/medmax/anole_7b_hf"
MODEL_NAME = "anole-7b"

# ──────────────────────── Request / Response schemas ────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    n: int = 1
    stop: list[str] | str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ══════════════════════════════════════════════════════════════════
#  GPU Worker  (one per GPU, runs as a subprocess)
# ══════════════════════════════════════════════════════════════════

MODEL_TOKENS_RE = re.compile(r"\[MODEL_TOKENS\]\s*([\s\S]*?)\s*\[/MODEL_TOKENS\]")
EOS_TOKEN_ID = 2


def run_worker(gpu_id: int, port: int, model_path: str):
    """Start a single-GPU inference worker on the given port."""
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    _model = None
    _tokenizer = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _model, _tokenizer
        from transformers import ChameleonForConditionalGeneration, AutoTokenizer

        print(f"[Worker GPU {gpu_id}] Loading model from {model_path} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _model = ChameleonForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        _model.eval()
        print(f"[Worker GPU {gpu_id}] Model loaded, port {port}")
        yield
        del _model, _tokenizer

    worker_app = FastAPI(lifespan=lifespan)

    @worker_app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"{msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(msg.content)
            elif msg.role == "assistant":
                prompt_parts.append(msg.content)
        prompt = "\n".join(prompt_parts)

        m = MODEL_TOKENS_RE.search(prompt)
        if m:
            token_str = m.group(1).strip()
            token_ids = [int(t) for t in re.split(r"[,\s]+", token_str) if t.strip()]
            while token_ids and token_ids[-1] == EOS_TOKEN_ID:
                token_ids.pop()
            after_text = prompt[m.end():].strip()
            if after_text:
                extra_ids = _tokenizer.encode(after_text, add_special_tokens=False)
                token_ids.extend(extra_ids)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
            prompt_len = input_ids.shape[1]
        else:
            inputs = _tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = _model.generate(
                input_ids=input_ids,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01),
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

        new_tokens = outputs[0][prompt_len:]
        response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(new_tokens)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                total_tokens=prompt_len + completion_tokens,
            ),
        )

    @worker_app.get("/health")
    async def health():
        return {"status": "ok", "gpu": gpu_id}

    uvicorn.run(worker_app, host="127.0.0.1", port=port, log_level="warning")


# ══════════════════════════════════════════════════════════════════
#  Load Balancer  (main process, round-robin across workers)
# ══════════════════════════════════════════════════════════════════

worker_urls: list[str] = []
_rr_counter = 0


@asynccontextmanager
async def lb_lifespan(app: FastAPI):
    """Wait until all GPU workers are ready."""
    print(f"[LB] Waiting for {len(worker_urls)} GPU workers ...")
    for url in worker_urls:
        for attempt in range(300):
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{url}/health", timeout=2)
                    if r.status_code == 200:
                        print(f"[LB] Worker {url} ready")
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            print(f"[LB] WARNING: Worker {url} not ready after 5 min")
    print(f"[LB] All workers ready. Load balancer on port {app.state.port}")
    yield


lb_app = FastAPI(lifespan=lb_lifespan)


@lb_app.post("/v1/chat/completions")
async def lb_chat(request: ChatCompletionRequest):
    global _rr_counter
    url = worker_urls[_rr_counter % len(worker_urls)]
    _rr_counter += 1

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json=request.model_dump(),
        )
        return resp.json()


@lb_app.get("/v1/models")
async def lb_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@lb_app.get("/health")
async def lb_health():
    return {"status": "ok", "workers": len(worker_urls)}


# ══════════════════════════════════════════════════════════════════
#  Main: spawn workers + load balancer
# ══════════════════════════════════════════════════════════════════


def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Multi-GPU Anole inference server")
    parser.add_argument("--port", type=int, default=8000, help="Load balancer port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5", help="Comma-separated GPU IDs")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--worker_base_port", type=int, default=8010, help="First worker port")
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"[Main] Starting {len(gpu_ids)} workers on GPUs: {gpu_ids}")

    global worker_urls
    workers = []
    for i, gpu_id in enumerate(gpu_ids):
        wport = args.worker_base_port + i
        worker_urls.append(f"http://127.0.0.1:{wport}")
        p = mp.Process(
            target=run_worker,
            args=(gpu_id, wport, args.model_path),
            daemon=True,
        )
        p.start()
        workers.append(p)
        print(f"[Main] Worker GPU {gpu_id} -> port {wport} (PID {p.pid})")

    def cleanup(sig, frame):
        print("\n[Main] Shutting down workers ...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    lb_app.state.port = args.port
    uvicorn.run(lb_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
