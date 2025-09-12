import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

model_id = r"/home/st426/system/global_graph/Biomistral-Calme-Instruct-7b"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

device_map = {"": "cuda:0"}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print("hf_device_map:", getattr(model, "hf_device_map", None))
print("any param device:", next(model.parameters()).device)
print("cuda available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")


dataset = load_dataset("json", data_files="/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換前/train.json")
print(dataset)



peft_config = LoraConfig(
    r=16,   
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ========= 訓練設定 =========
training_args = TrainingArguments(
    output_dir="./lora-biomistral-neural-invasion",
    per_device_train_batch_size=1,   # batch_size=1
    gradient_accumulation_steps=4,   # 累積 4 次再更新
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
)

# ========= 格式轉換 =========
def formatting_func(example):
    text = f"{example['instruction']}\n\nInput:\n{example['input']}\n\nOutput:\n{example['output']}"
    return text   



# ========= Trainer =========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,
    peft_config=peft_config,
)

trainer.train()

# ========= Save =========
trainer.model.save_pretrained("data/Neural_Invasion_Detection轉換前/lora-biomistral-neural-invasion")
tokenizer.save_pretrained("data/Neural_Invasion_Detection轉換前/lora-biomistral-neural-invasion")
