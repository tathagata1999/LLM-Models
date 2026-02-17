import modal
from modal import Volume, Image

# -----------------------
# Modal App + Image
# -----------------------
app = modal.App("pricer-service")

image = (
    Image.debian_slim()
    .pip_install(
        "huggingface_hub",
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
    )
)

secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"

# -----------------------
# Model + LoRA config
# -----------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"   # must match LoRA base
HF_USER = "Tatha1999"
PROJECT_NAME = "llama3.1_model_price"
RUN_NAME = "2026-02-16_15.41.59-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "5f4c36aa147100a5b100417fa0a408137971f171"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

# HF cache volume → avoids repeated downloads
CACHE_DIR = "/cache"
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# Keep 1 warm container for low latency
MIN_CONTAINERS = 1

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


# -----------------------
# GPU Actor (loads model once)
# -----------------------
@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        torch.set_float32_matmul_precision("medium")

        # 4-bit quantization (T4 safe)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Base model (4-bit)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Disable KV cache → saves 2–3 GB VRAM on T4
        self.base_model.config.use_cache = False

        # Attach LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION,
        )

        self.model.eval()

    # -----------------------
    # Single price prediction
    # -----------------------
    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch

        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # prevents long-text OOM
        ).to(self.base_model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if PREFIX in decoded:
            decoded = decoded.split(PREFIX)[-1]

        decoded = decoded.replace(",", "")

        match = re.search(r"[-+]?\d*\.\d+|\d+", decoded)
        return float(match.group()) if match else 0.0

    # -----------------------
    # Batch pricing (much faster & cheaper)
    # -----------------------
    @modal.method()
    def price_batch(self, descriptions: list[str]) -> list[float]:
        return [self.price(d) for d in descriptions]


# -----------------------
# Local test entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    pricer = Pricer()

    items = [
        "Quadcast HyperX condenser mic USB-C crystal clear audio",
        "Apple iPhone 13 128GB new",
        "Sony WH-1000XM4 wireless headphones",
    ]

    results = pricer.price_batch.remote(items)

    for item, price in zip(items, results):
        print(f"{item} → ${price}")
