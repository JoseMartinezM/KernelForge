import json
from typing import Any

import aiohttp
import modal
import socket
import subprocess

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04", add_python="3.13")
        .entrypoint([])
        .uv_pip_install("vllm==0.21.0")
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "VLLM_LOGS_STATS_INTERVAL": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
            }
        )
)

MODEL_NAME = "google/gemma-4-E4B-it"
MODEL_REVISION = "d6436b3d62967e1af08bbb046c6300b2a9ae8e85"

SPECULATIVE_MODEL_NAME = "google/gemma-4-E4B-it-assistant"
SPECULATIVE_MODEL_REVISION = "4a5c666f89be588c72e0b3a9b49c118513cedff6"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000
FAST_BOOT = False

with vllm_image.imports():
    import requests

def sleep(level=1):
    requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level={level}"
    ).raise_for_status()


def wake_up():
    requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()

def wait_ready(proc: subprocess.Popen[bytes]):
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Magnets!!! How do they work????"}],
        "max_tokens": 16,
    }

    for _ in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()


@app.cls(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=3 * MINUTES,  # how long should we stay up with no requests?
    timeout=20 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=8,
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
            "--uvicorn-log-level=info",
            "--async-scheduling",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.97",
            "--enable-sleep-mode",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "16384",
            "--compilation-config",
            json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8]}),
        ]
        # enforce-eager disables both Torch compilation and CUDA graph capture
        # default is no-enforce-eager. see the --compilation-config flag for tighter control
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # add model-specific configuration
        cmd += [
            # skip multimedia support, just language
            "--limit-mm-per-prompt",
            json.dumps({'image': 0, 'video': 0, 'audio': 0}),
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
async def test(test_timeout=15 * MINUTES, content=None, twice=True):
    url = await VllmServer().serve.get_web_url.aio()

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

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
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
