from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

# Load X-Sum dataset and randomly sample 1000 entries
dataset = load_dataset('xsum', split='train')
sampled_dataset = dataset.shuffle(seed=42).select(range(1000))

# Load GPT-NeoX model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b')
model.eval()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Save results to a text file
with open('xsum_continuations.txt', 'w', encoding='utf-8') as file:
    for sample in sampled_dataset:
        input_text = sample['document']
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        
        # Generate continuation text
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Write to file
        file.write(f"Original: {input_text}\nGenerated: {generated_text}\n\n")