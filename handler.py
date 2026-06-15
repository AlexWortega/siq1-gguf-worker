import os, json, time, subprocess, requests, runpod
from huggingface_hub import hf_hub_download

PORT = 8080
# Production sampling defaults (piagent src/engine/llama.ts — soft penalties keep temp 0.1 from
# looping / garbling tool-call JSON). Applied unless the client overrides; tunable via SAMPLING_DEFAULTS.
SAMPLING_DEFAULTS = json.loads(os.environ.get("SAMPLING_DEFAULTS",
    '{"temperature":0.6,"top_p":0.9,"top_k":40,"min_p":0,"repeat_penalty":1.1,"presence_penalty":0.4}'))
def boot():
    print("[worker] downloading GGUF", flush=True)
    path = hf_hub_download(os.environ["GGUF_REPO"], os.environ["GGUF_FILE"],
                           token=os.environ.get("HF_TOKEN"))
    print("[worker] starting llama-server", flush=True)
    subprocess.Popen(["/app/llama-server", "-m", path, "-ngl", os.environ.get("NGL","99"),
                      "-c", os.environ.get("N_CTX","16384"), "--host", "127.0.0.1",
                      "--port", str(PORT), "--jinja"])
    for _ in range(600):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                print("[worker] llama-server ready", flush=True); return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready")

boot()

def handler(job):
    inp = job.get("input", {})
    # passthrough ALL OpenAI params (top_p, top_k, min_p, repeat_penalty, presence_penalty,
    # frequency_penalty, stop, chat_template_kwargs, ...) to llama-server — production sampling matters.
    body = dict(inp)
    if not body.get("messages"):
        body["messages"] = [{"role": "user", "content": body.get("prompt", "")}]
    body.setdefault("max_tokens", 1024)
    for k, v in SAMPLING_DEFAULTS.items():   # hoisted prod sampling — applied unless client overrides
        body.setdefault(k, v)
    body["stream"] = False
    body.pop("prompt", None)
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=900)
    return r.json()

runpod.serverless.start({"handler": handler})
