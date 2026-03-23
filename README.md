# LocalLLM — Inference Benchmarking

Benchmark LLM inference across three approaches to compare speed, cost, and complexity tradeoffs.

## The Three Approaches

| Approach | Provider | Hardware | Setup |
|----------|----------|----------|-------|
| **Local** | LM Studio | Your machine | Download model, run locally |
| **Managed** | Groq | Groq cloud | API key, zero setup |
| **Self-hosted cloud** | Modal + vLLM | A10G GPU | Deploy serverless, pay per use |

All three run **Qwen 2.5 7B Instruct** on the same test prompt (hash table explanation, 512 token max).

## What It Measures

- **Tokens per second** — raw throughput
- **Total latency** — end-to-end response time
- **Token counts** — prompt and completion tokens
- **Comparative analysis** — side-by-side results across all providers

## Setup

```bash
uv sync
```

### Environment Variables

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_key
MODAL_TOKEN_ID=your_id
MODAL_TOKEN_SECRET=your_secret
```

### Local (LM Studio)

1. Install [LM Studio](https://lmstudio.ai/)
2. Download Qwen 2.5 7B Instruct
3. Start the local server

### Cloud (Modal)

Deploy the vLLM server to Modal:

```bash
uv run modal deploy modal_llm.py
```

This provisions an A10G GPU with a 5-minute scaledown window and persistent model caching.

## Usage

Run the benchmark:

```bash
uv run python benchmark.py
```

## Project Structure

```
├── benchmark.py     # Main benchmarking script — compares all 3 providers
├── modal_llm.py     # Modal app serving vLLM on A10G GPU
├── modal_test.py    # Simple vLLM test deployment
├── main.py          # Entry point
├── pyproject.toml   # Dependencies
└── .env             # API keys (not committed)
```

## Tech Stack

- **LiteLLM** — unified API wrapper across providers
- **Modal** — serverless GPU compute
- **vLLM** — high-throughput inference server
- **OpenAI SDK** — API client
- **psutil** — system monitoring

## Key Takeaway

Each approach has a different sweet spot: local is free but hardware-limited, managed is fast but costs per token, self-hosted cloud gives you control with pay-per-use GPU. This benchmark gives you real numbers to compare.
