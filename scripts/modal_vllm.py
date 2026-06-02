import asyncio
import json
import os
import subprocess
import time
from typing import Any

import aiohttp
import modal

VLLM_VERSION = os.environ.get("KF_MODAL_VLLM_VERSION", "0.21.0")

vllm_image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    .run_commands("ln -sf $(which python3) /usr/bin/python")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "OMP_NUM_THREADS": "1",
            "TORCH_CPP_LOG_LEVEL": "FATAL",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_MEDIA_LOADING_THREAD_COUNT": "1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

MODEL_NAME = "google/gemma-4-E4B-it"
MODEL_REVISION = "d6436b3d62967e1af08bbb046c6300b2a9ae8e85"

SPECULATIVE_MODEL_NAME = "google/gemma-4-E4B-it-assistant"
SPECULATIVE_MODEL_REVISION = "4a5c666f89be588c72e0b3a9b49c118513cedff6"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("kernelforge-vllm")

N_GPU = 1
GPU_TYPE = os.environ.get("KF_MODAL_GPU", "L4")
MINUTES = 60  # seconds
VLLM_PORT = 8000
MAX_INPUTS = int(os.environ.get("KF_MODAL_MAX_INPUTS", "4"))
MAX_MODEL_LEN = int(os.environ.get("KF_MODAL_MAX_MODEL_LEN", "12288"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("KF_MODAL_GPU_MEMORY_UTILIZATION", "0.82"))
MIN_CONTAINERS = int(os.environ.get("KF_MODAL_MIN_CONTAINERS", "0"))
WARMUP_REQUESTS = int(os.environ.get("KF_MODAL_WARMUP_REQUESTS", "1"))

with vllm_image.imports():
    import requests


def sleep(level=1):
    requests.post(
        f"http://127.0.0.1:{VLLM_PORT}/sleep?level={level}"
    ).raise_for_status()


def wake_up():
    requests.post(f"http://127.0.0.1:{VLLM_PORT}/wake_up").raise_for_status()


def check_running(proc: subprocess.Popen[bytes]) -> None:
    if (return_code := proc.poll()) is not None:
        raise RuntimeError(f"vLLM exited with {return_code}")


def wait_ready(proc: subprocess.Popen[bytes], timeout: int = 10 * MINUTES) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            check_running(proc)
            requests.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5).raise_for_status()
            return
        except (requests.RequestException, RuntimeError):
            check_running(proc)
            time.sleep(2)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Magnets!!! How do they work????"}],
        "max_tokens": 16,
    }

    for _ in range(WARMUP_REQUESTS):
        requests.post(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=120,
        ).raise_for_status()


def modal_proxy_headers(
    *, proxy_key: str | None = None, proxy_secret: str | None = None, required: bool = True
) -> dict[str, str]:
    modal_key = (
        proxy_key
        or os.environ.get("KF_MODAL_PROXY_KEY")
        or os.environ.get("MODAL_API_KEY")
        or os.environ.get("MODAL_API_TOKEN")
    )
    modal_secret = (
        proxy_secret or os.environ.get("KF_MODAL_PROXY_SECRET") or os.environ.get("MODAL_API_SECRET")
    )
    if not modal_key or not modal_secret:
        if required:
            raise RuntimeError(
                "Set KF_MODAL_PROXY_KEY or MODAL_API_KEY or MODAL_API_TOKEN, plus "
                "KF_MODAL_PROXY_SECRET or MODAL_API_SECRET, to call the "
                "proxy-authenticated Modal endpoint."
            )
        return {}
    return {"Modal-Key": modal_key, "Modal-Secret": modal_secret}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    cpu=6.0,
    memory=32768,
    scaledown_window=1 * MINUTES,  # keep idle GPU time short for research runs
    timeout=15 * MINUTES,  # allow model download/first snapshot build on cold deploys
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=MIN_CONTAINERS,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=MAX_INPUTS,
)
class VllmServer:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=warning",
            "--generation-config",
            "vllm",
            "--dtype",
            "bfloat16",
            "--optimization-level",
            "0",
            "--performance-mode",
            "interactivity",
            "--async-scheduling",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--enable-sleep-mode",
            "--enforce-eager",
            "--max-num-seqs",
            str(MAX_INPUTS),
            "--max-num-batched-tokens",
            str(MAX_MODEL_LEN),
            "--max-num-partial-prefills",
            "1",
            "--language-model-only",
            "--mm-processor-cache-gb",
            "0",
            "--skip-mm-profiling",
            "--no-enable-log-requests",
            "--structured-outputs-config.backend=xgrammar",
            "--structured-outputs-config.enable_in_reasoning=False",
        ]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # add model-specific configuration
        cmd += [
            # enable reasoning and tool use
            # "--enable-auto-tool-choice",
            "--reasoning-parser",
            "gemma4",
            # "--tool-call-parser gemma4",
        ]

        # add speculative decoding
        # cmd += [
        #     "--speculative-config",
        #     f"'{json.dumps({'model': SPECULATIVE_MODEL_NAME, 'revision': SPECULATIVE_MODEL_REVISION, 'num_speculative_tokens': 2})}'",
        # ]

        print(*cmd)

        self.vllm_proc = subprocess.Popen(cmd)
        wait_ready(self.vllm_proc)
        warmup()
        sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        wake_up()
        wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()


@app.local_entrypoint()
async def test(
    test_timeout=15 * MINUTES,
    content=None,
    twice=True,
    proxy_key=None,
    proxy_secret=None,
):
    twice = parse_bool(twice)
    print("Resolving Modal web URL", flush=True)
    url = await asyncio.to_thread(VllmServer().serve.get_web_url)
    print(f"Resolved Modal web URL: {url}", flush=True)
    headers = modal_proxy_headers(proxy_key=proxy_key, proxy_secret=proxy_secret)

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        print(f"Running health check for server at {url}")
        health_deadline = time.monotonic() + test_timeout - 1 * MINUTES
        last_status: int | str | None = None
        while True:
            try:
                async with session.get("/health", timeout=30) as resp:
                    if resp.status == 200:
                        break
                    status = resp.status
                    if status in {401, 403}:
                        text = await resp.text()
                        raise RuntimeError(f"Modal proxy authentication failed: {status} {text}")
            except (TimeoutError, aiohttp.ClientError):
                status = "timeout"
            if status != last_status:
                print(f"Health check not ready yet: {status}", flush=True)
                last_status = status
            if time.monotonic() >= health_deadline:
                raise AssertionError(f"Failed health check for server at {url}: {status}")
            await asyncio.sleep(5)
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    # explicitly enable thinking for this model
    payload["chat_template_kwargs"] = {"enable_thinking": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if content:
                print(content, end="")
            else:
                print("\n", chunk)
    print()
