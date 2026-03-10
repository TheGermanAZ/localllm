import modal
import subprocess

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "/models"

volume = modal.Volume.from_name("model-cache", create_if_missing=True)

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


@app.function(
    gpu="A10G",
    volumes={MODEL_DIR: volume},
    timeout=600,
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    """Download model if needed, then start vLLM server."""
    import os
    model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_NAME, local_dir=model_path)
        volume.commit()

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    subprocess.Popen(cmd)
