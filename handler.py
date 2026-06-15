import os, time, subprocess, requests, runpod
from huggingface_hub import hf_hub_download

PORT = 8080
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
    body = {"messages": inp.get("messages", [{"role":"user","content": inp.get("prompt","")}]),
            "max_tokens": inp.get("max_tokens", 1024), "temperature": inp.get("temperature", 0.7),
            "stream": False}
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=900)
    return r.json()

runpod.serverless.start({"handler": handler})
