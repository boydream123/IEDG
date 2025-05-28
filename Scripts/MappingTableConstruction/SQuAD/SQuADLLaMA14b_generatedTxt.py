from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch


# 加载SQuAD数据集
squad_dataset = load_dataset("squad")

# 抽取1000个样本
samples = squad_dataset['train'].shuffle(seed=42).select(range(1000))

# 加载LLaMA-7B模型及分词器
tokenizer = AutoTokenizer.from_pretrained('facebook/llama-14b')
model = AutoModelForCausalLM.from_pretrained('facebook/llama-14b')
model.eval()

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 续写示例
def generate_continuation(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return continuation

# 对每个样本进行续写
for sample in samples:
    context = sample['context']
    question = sample['question']
    text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    continuation = generate_continuation(text)
    print(continuation)