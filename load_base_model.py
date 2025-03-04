from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

model_path="ibm-granite/granite-3.2-2b-instruct"
device= "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

model.save_pretrained("base_models/granite-3.2-2b-instruct2")