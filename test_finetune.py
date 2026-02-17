from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Configuration for RTX 3090 (24GB VRAM)
max_seq_length = 2048 # 2048 is safe. You can likely push to 4096 on a 3090 with Llama-3-8B.
dtype = None # None = auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+ (like your 3090)
load_in_4bit = True # Essential for 24GB VRAM

# 2. Load the Model (Switched to Llama-3 8B for best performance/size balance)
model_name = "unsloth/llama-3-8b-bnb-4bit" 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Add your HF token if using gated models like Llama-3
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized to 0
    bias = "none",    # Optimized to "none"
    use_gradient_checkpointing = "unsloth", # "unsloth" saves 30% VRAM!
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Load Dataset
# Note: Ensure your dataset has a "text" field or use a formatting function.
# Here we stick to your LAION OIG link.
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

# 5. Initialize Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # <--- CRITICAL: Tells trainer which column to use
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    
    args = SFTConfig(
        per_device_train_batch_size = 2, # Safe for 3090
        gradient_accumulation_steps = 4, # Effective batch size = 2 * 4 = 8
        warmup_steps = 5,
        max_steps = 60, # Increase this for a real run (e.g., 500+)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # 3090 supports BF16, so this will be True
        logging_steps = 1,
        optim = "adamw_8bit", # Saves VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Train
trainer_stats = trainer.train()

print("Training complete!")

# 7. (Optional) Save the model
# model.save_pretrained("lora_model") 
# tokenizer.save_pretrained("lora_model")