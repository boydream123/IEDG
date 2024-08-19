import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# 加载BART模型和tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')

# 准备数据集
def preprocess_function(examples):
    inputs = [doc for doc in examples['document']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # 设置“labels”字段
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained('./fine-tuned-bart')
tokenizer.save_pretrained('./fine-tuned-bart')