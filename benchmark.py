import os
import time
import litellm

PROMPT = "Explain what a hash table is, how it handles collisions, and give a Python example."

MODAL_URL = os.environ.get("MODAL_URL", "http://localhost:8000/v1")


def benchmark(model, label, **kwargs):
    """Time a single completion and measure tokens/second."""
    try:
        start = time.time()
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=512,
            **kwargs,
        )
        elapsed = time.time() - start
        output = response.choices[0].message.content
        tokens = response.usage.completion_tokens
        tps = tokens / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Tokens generated:  {tokens}")
        print(f"  Total time:        {elapsed:.2f}s")
        print(f"  Tokens/sec:        {tps:.1f}")
        print(f"  First 100 chars:   {output[:100]}...")
        return {"label": label, "tokens": tokens, "time": elapsed, "tps": tps}
    except Exception as e:
        print(f"\n  SKIP: {label} — {e}")
        return None


results = []

def run(result):
    if result:
        results.append(result)

# --- 1. Local (Ollama) ---
run(benchmark("ollama/llama3.1:8b", "LOCAL (Ollama on your machine)"))

# --- 2. Self-hosted cloud (Modal + vLLM) ---
run(benchmark(
    "openai/Qwen/Qwen2.5-7B-Instruct",
    "MODAL (your vLLM on A10G GPU)",
    api_base=MODAL_URL,
    api_key="not-needed",
))

# --- 3. Managed provider (Groq) ---
run(benchmark("groq/llama-3.1-8b-instant", "GROQ (managed inference provider)"))

# --- Comparison table ---
if results:
    print(f"\n{'='*60}")
    print(f"  COMPARISON ({len(results)} provider{'s' if len(results) != 1 else ''})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label']:<40} {r['tps']:>8.1f} tok/s  {r['time']:>6.2f}s total")

    if len(results) >= 2:
        fastest = max(results, key=lambda r: r["tps"])
        slowest = min(results, key=lambda r: r["tps"])
        print(f"\n  Fastest: {fastest['label']}")
        print(f"  {fastest['tps'] / slowest['tps']:.1f}x faster than {slowest['label']}")
else:
    print("\n  No providers were reachable.")
