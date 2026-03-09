import modal
import subprocess

# Define the container image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("llama-inference", image=image)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Persistent volume to cache model weights across cold starts
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(
    gpu="A10G",  # 24GB VRAM, plenty for an 8B model
    volumes={"/models": volume},
    timeout=600,
)
@modal.web_server(port=8000)
def serve():
    """Spawn a vLLM server serving Llama 3.1 8B."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--download-dir", "/models",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    subprocess.Popen(cmd)