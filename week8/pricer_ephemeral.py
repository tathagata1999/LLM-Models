# import modal
# from modal import Image

# # Setup

# app = modal.App("price prediction")
# image = Image.debian_slim().pip_install(
#     "torch", "transformers", "bitsandbytes", "accelerate", "peft"
# )
# secrets = [modal.Secret.from_name("huggingface-secret")]

# # Constants

# GPU = "T4"
# BASE_MODEL = "meta-llama/Llama-3.1-8B"
# PROJECT_NAME = "llama3.1_model_price"
# HF_USER = "Tatha1999"  # your HF name here! Or use mine if you just want to reproduce my results.
# #RUN_NAME = "2025-11-28_18.47.07"
# RUN_NAME = "2026-02-16_15.41.59-lite"
# REVISION = "5f4c36aa147100a5b100417fa0a408137971f171"
# PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
# #REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
# FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


# @app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
# def price(description: str) -> float:
#     import re
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
#     from peft import PeftModel

#     PREFIX = "Price is $"
#     QUESTION = "What does this cost to the nearest dollar?"

#     prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

#     # Quant Config
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )

#     # Load model and tokenizer

#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL, quantization_config=quant_config, device_map="auto"
#     )

#     fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

#     set_seed(42)
#     inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         outputs = fine_tuned_model.generate(inputs, max_new_tokens=5)
#     result = tokenizer.decode(outputs[0])
#     contents = result.split("Price is $")[1]
#     contents = contents.replace(",", "")
#     match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
#     return float(match.group()) if match else 0

import modal
from modal import Volume, Image

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

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # MUST match LoRA base
HF_USER = "Tatha1999"
PROJECT_NAME = "llama3.1_model_price"
RUN_NAME = "2026-02-16_15.41.59-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "5f4c36aa147100a5b100417fa0a408137971f171"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

CACHE_DIR = "/cache"
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

MIN_CONTAINERS = 0

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


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

        # Reduce CUDA fragmentation
        torch.set_float32_matmul_precision("medium")

        # 4-bit quant config (T4 optimized)
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

        # Base model (FP16 compute, 4-bit weights)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Disable KV cache globally to save VRAM on T4
        self.base_model.config.use_cache = False

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION,
        )

        # Set eval mode
        self.model.eval()

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch

        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        # Memory-safe tokenization with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.base_model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4,          # small output → saves VRAM
                do_sample=False,
                temperature=0.0,
                use_cache=False,           # critical for T4 memory
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract numeric price
        if PREFIX in decoded:
            decoded = decoded.split(PREFIX)[-1]

        decoded = decoded.replace(",", "")

        match = re.search(r"[-+]?\d*\.\d+|\d+", decoded)
        return float(match.group()) if match else 0.0
