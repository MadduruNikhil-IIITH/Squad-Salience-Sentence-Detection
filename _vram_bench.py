import gc
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

candidates = [
    ("microsoft/Phi-4-mini-instruct", "4bit"),
    ("Qwen/Qwen2.5-7B-Instruct", "4bit"),
    ("microsoft/Phi-4-mini-instruct", "16bit"),
]

prompt = "Generate two short extractive QA pairs from this passage: The Pacific Ocean covers more than 30% of the Earth's surface."

print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(0)

for model_id, precision in candidates:
    print("\n" + "=" * 80)
    print(f"TEST model={model_id} precision={precision}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    tok = None
    mdl = None
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if precision == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.65,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        curr_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        curr_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print("status=OK")
        print(f"peak_alloc_gb={peak_alloc:.3f}")
        print(f"peak_reserved_gb={peak_reserved:.3f}")
        print(f"curr_alloc_gb={curr_alloc:.3f}")
        print(f"curr_reserved_gb={curr_reserved:.3f}")
        print(f"sample_gen={gen[:160].replace(chr(10), ' ')}")

    except Exception as e:
        print(f"status=ERROR type={type(e).__name__}")
        print(f"message={str(e)}")
        traceback.print_exc(limit=1)
    finally:
        del mdl
        del tok
        gc.collect()
        torch.cuda.empty_cache()
