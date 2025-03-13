import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from deepseek_vl2.utils.io import load_pil_images 
from typing import Dict, Tuple, List, Literal, Optional

# Load model and processor
model_path = "deepseek-ai/deepseek-vl2"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model = vl_gpt.to(torch.bfloat16).cuda()

# Define PEFT configuration (LoRA)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# System prompt (customize accordingly)
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# Fix format_data function
def format_data(sample: Dict[str, str]) -> List[Dict[str, str]]:
    formatted_sample: List[Dict[str, str]] = [{
        "role": "<|User|>", 
        "content": f"{sample['query']} <image>",
        "images": sample['image']  # Fixed the missing comma here
    }, {
        "role": "<|Assistant|>", 
        "content": f"{sample['label'][0]}"
    }]
    return formatted_sample

# Load dataset
dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:10%]", "val[:10%]", "test[:10%]"])

# Apply data formatting
train_dataset = [format_data(sample) for sample in train_dataset if format_data(sample) is not None]
eval_dataset = [format_data(sample) for sample in eval_dataset if format_data(sample) is not None]
test_dataset = [format_data(sample) for sample in test_dataset if format_data(sample) is not None]
print(f"First training sample after processing: {train_dataset[0]}")

from trl import SFTConfig
# Configure training arguments
training_args = SFTConfig(
    output_dir="deepseekvl2_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    push_to_hub=False,
    remove_unused_columns=False,
    dataset_text_field=None,  # explicitly set to None
    dataset_kwargs={"skip_prepare_dataset": True},  # add this line
    report_to="wandb",
)

# Define collate_fn to process batches
def collate_fn(examples):
    batch_conversations = []
    batch_images = []

    for example in examples:
        if example is None:
            print("empty sample")
            continue

        print(f"Example: {example}") 
        
        # Extract user and assistant messages
        formatted_sample = example  # Already formatted in format_data
        
        # Append the user and assistant conversations
        batch_conversations.append({
            "role": formatted_sample[0]["role"],
            "content": formatted_sample[0]["content"],
        })
        batch_conversations.append({
            "role": formatted_sample[1]["role"],
            "content": formatted_sample[1]["content"],
        })

        # Append image to batch_images
        # if formatted_sample[0]["images"].mode != 'RGB':
        #      formatted_sample[0]["images"].convert('RGB')
        batch_images.append(formatted_sample[0]["images"].convert('RGB'))  # User message contains the image

    print(len(batch_conversations))
    print(len(batch_images))
    if not batch_conversations or not batch_images:
        print("Warning: Empty conversations or images batch!")

    # Pass both conversations and images to the processor
    inputs = vl_chat_processor(
        conversations=batch_conversations,  # Provide the batch of conversations
        images=batch_images,  # Provide the batch of images
        return_tensors="pt",
        padding=True,
        force_batchify=True,
        system_prompt=system_message
    )

    # Create labels for the inputs, ensuring padding is handled correctly
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Handle special tokens in the labels
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(tok) for tok in ['<image>', '<|User|>', '<|Assistant|>']
    ]

    for token_id in special_token_ids:
        labels[labels == token_id] = -100

    inputs["labels"] = labels
    return inputs

# Setup trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)

# Start training
trainer.train()

# Save trained model
trainer.save_model(training_args.output_dir)

# Clear memory after training (optional)
import gc
gc.collect()
torch.cuda.empty_cache()

print("Training completed and model saved.")
