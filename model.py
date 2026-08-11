"""
model.py — Model wrapper for the synergy expansion experiment.

Wraps Ollama's local API to provide a consistent interface across all 20
models in the roster. Enforces locked-in experimental settings:
  - temperature = 0
  - seed = 42
  - no retries on failure (failures are logged and surfaced, not silently retried)

Usage:
    from model import ModelWrapper

    wrapper = ModelWrapper("qwen2.5:7b")
    response = wrapper.generate("What is 2 + 2?")
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# Locked experimental settings — do not change without updating protocol_v1.md
TEMPERATURE = 0
SEED = 42
NUM_PREDICT = 2048  # max tokens for response; adjust if math/coding answers get truncated

# Final 15-model roster (all <=9B params, Q4_K_M quantization via Ollama)
# Locked for expanded experiment: 15 models × 100 questions (60 math, 40 coding)
MODEL_ROSTER = [
    "yi:6b",
    "deepseek-coder:6.7b",
    "phi3:mini",
    "falcon3:3b",
    "llama3.2:3b",
    "gemma2:2b",
    "smollm2:1.7b",
    "tinyllama:1.1b",
    "qwen2.5:0.5b",
    "llama3:8b",
    "mistral:7b",
    "phi3.5:3.8b",
    "gemma2:9b",
    "qwen2.5:7b",
    "llama3.2:1b",
]


@dataclass
class ModelResponse:
    model_name: str
    prompt: str
    output_text: str
    success: bool
    error_message: Optional[str]
    latency_seconds: float
    raw_response: Optional[dict]


class ModelWrapper:
    """Wraps a single Ollama model with locked experimental settings."""

    def __init__(self, model_name: str, timeout_seconds: int = 120):
        if model_name not in MODEL_ROSTER:
            raise ValueError(
                f"'{model_name}' is not in MODEL_ROSTER. "
                f"Add it to model.py's MODEL_ROSTER if this is intentional."
            )
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> ModelResponse:
        """
        Send a single prompt to the model and return a structured response.
        No retries — a failure is recorded as such and returned immediately,
        per protocol_v1.md.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": NUM_PREDICT,
            },
        }

        start_time = time.time()
        try:
            resp = requests.post(
                GENERATE_ENDPOINT, json=payload, timeout=self.timeout_seconds
            )
            latency = time.time() - start_time

            if resp.status_code != 200:
                return ModelResponse(
                    model_name=self.model_name,
                    prompt=prompt,
                    output_text="",
                    success=False,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    latency_seconds=latency,
                    raw_response=None,
                )

            data = resp.json()
            output_text = data.get("response", "")

            return ModelResponse(
                model_name=self.model_name,
                prompt=prompt,
                output_text=output_text,
                success=True,
                error_message=None,
                latency_seconds=latency,
                raw_response=data,
            )

        except requests.exceptions.Timeout:
            latency = time.time() - start_time
            return ModelResponse(
                model_name=self.model_name,
                prompt=prompt,
                output_text="",
                success=False,
                error_message=f"Timeout after {self.timeout_seconds}s",
                latency_seconds=latency,
                raw_response=None,
            )

        except requests.exceptions.ConnectionError as e:
            latency = time.time() - start_time
            return ModelResponse(
                model_name=self.model_name,
                prompt=prompt,
                output_text="",
                success=False,
                error_message=f"Connection error (is Ollama running?): {e}",
                latency_seconds=latency,
                raw_response=None,
            )

        except Exception as e:
            latency = time.time() - start_time
            return ModelResponse(
                model_name=self.model_name,
                prompt=prompt,
                output_text="",
                success=False,
                error_message=f"Unexpected error: {type(e).__name__}: {e}",
                latency_seconds=latency,
                raw_response=None,
            )


def check_model_available(model_name: str) -> bool:
    """Check whether a model is currently pulled and available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code != 200:
            return False
        available = [m["name"] for m in resp.json().get("models", [])]
        return model_name in available
    except requests.exceptions.ConnectionError:
        return False


def check_all_roster_models() -> dict:
    """Check availability of every model in MODEL_ROSTER. Returns {name: bool}."""
    return {name: check_model_available(name) for name in MODEL_ROSTER}


if __name__ == "__main__":
    # Quick smoke test: check which roster models are currently pulled
    print("Checking model availability against MODEL_ROSTER...\n")
    status = check_all_roster_models()
    for name, available in status.items():
        marker = "✓" if available else "✗ NOT PULLED"
        print(f"  {marker}  {name}")

    missing = [name for name, ok in status.items() if not ok]
    if missing:
        print(f"\n{len(missing)} model(s) not yet pulled. Run: ollama pull <name>")
    else:
        print("\nAll roster models available.")
