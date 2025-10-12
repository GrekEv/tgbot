# 🎓 Руководство по дообучению моделей для RAG Support Bot

## 🎯 Цель
Научить локальную модель Ollama лучше отвечать на вопросы поддержки, используя ваши данные.

## 📋 Что нужно для дообучения

1. **Ollama установлен и работает**
2. **Python 3.8+** с библиотеками для ML
3. **Данные для обучения** (диалоги поддержки)
4. **Минимум 8GB RAM** (рекомендуется 16GB)

## 🚀 Подготовка данных

### Шаг 1: Создание структуры папок

```bash
mkdir -p data/training
mkdir -p data/models
mkdir -p scripts
```

### Шаг 2: Подготовка данных для обучения

Создайте файл `data/training/support_conversations.json`:

```json
{
  "conversations": [
    {
      "question": "Как сбросить пароль?",
      "answer": "Для сброса пароля перейдите на страницу входа и нажмите 'Забыли пароль?'. Введите ваш email и следуйте инструкциям в письме.",
      "category": "account"
    },
    {
      "question": "Не работает оплата",
      "answer": "Проверьте данные карты и попробуйте другой способ оплаты. Если проблема остается, обратитесь в банк.",
      "category": "payment"
    },
    {
      "question": "Как отменить заказ?",
      "answer": "Заказы можно отменить в течение 24 часов после оформления через личный кабинет или обратившись в поддержку.",
      "category": "orders"
    }
  ]
}
```

### Шаг 3: Создание Modelfile для кастомной модели

Создайте файл `Modelfile`:

```dockerfile
FROM llama3.2:8b

# Системный промпт для оператора поддержки
SYSTEM """Ты профессиональный оператор службы поддержки компании. 
Твоя задача - помогать клиентам решать их проблемы вежливо и эффективно.
Используй только информацию из контекста для ответов.
Отвечай на русском языке.
Будь вежливым, профессиональным и полезным."""

# Параметры модели
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 500
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
```

## 🔧 Методы дообучения

### Метод 1: Fine-tuning с помощью Ollama

1. **Создайте кастомную модель:**
```bash
ollama create support-bot -f Modelfile
```

2. **Протестируйте модель:**
```bash
ollama run support-bot "Как сбросить пароль?"
```

### Метод 2: LoRA дообучение (продвинутый)

Создайте файл `scripts/train_lora.py`:

```python
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# Загрузка модели
model_name = "meta-llama/Llama-3.2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Настройка LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Загрузка данных
def load_training_data():
    with open("data/training/support_conversations.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = []
    for conv in data["conversations"]:
        prompt = f"Вопрос: {conv['question']}\nОтвет: {conv['answer']}"
        conversations.append({"text": prompt})
    
    return conversations

# Подготовка данных для обучения
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512
    )

# Загрузка и подготовка данных
data = load_training_data()
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./data/models/support-bot-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Запуск обучения
print("Начинаем обучение...")
trainer.train()
print("Обучение завершено!")

# Сохранение модели
model.save_pretrained("./data/models/support-bot-lora")
tokenizer.save_pretrained("./data/models/support-bot-lora")
```

### Метод 3: RAG с дообученными эмбеддингами

Создайте файл `scripts/train_embeddings.py`:

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Загрузка модели для эмбеддингов
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Загрузка данных
def load_training_data():
    with open("data/training/support_conversations.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for conv in data["conversations"]:
        # Создаем позитивные пары (вопрос-ответ)
        examples.append(InputExample(
            texts=[conv['question'], conv['answer']], 
            label=1.0
        ))
        
        # Создаем негативные пары (вопрос-случайный ответ)
        if len(data["conversations"]) > 1:
            random_conv = data["conversations"][np.random.randint(0, len(data["conversations"]))]
            if random_conv != conv:
                examples.append(InputExample(
                    texts=[conv['question'], random_conv['answer']], 
                    label=0.0
                ))
    
    return examples

# Подготовка данных
train_examples = load_training_data()
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Настройка функции потерь
train_loss = losses.CosineSimilarityLoss(model)

# Дообучение
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path='./data/models/support-embeddings'
)

print("Дообучение эмбеддингов завершено!")
```

## 🚀 Интеграция дообученной модели

### Обновление app_ollama.py

Добавьте функцию для работы с дообученной моделью:

```python
def call_finetuned_ollama(question, context=""):
    """Вызывает дообученную модель Ollama"""
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Используем дообученную модель
        model_name = "support-bot"  # Имя вашей дообученной модели
        
        prompt = f"""Контекст: {context}

Вопрос клиента: {question}

Ответ:"""
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
        
    except Exception as e:
        return f"Ошибка при обращении к дообученной модели: {str(e)}"
```

## 📊 Оценка качества модели

### Создание тестового набора

Создайте файл `data/training/test_set.json`:

```json
{
  "test_cases": [
    {
      "question": "Забыл пароль от аккаунта",
      "expected_keywords": ["сброс", "пароль", "email", "инструкции"],
      "category": "account"
    },
    {
      "question": "Не приходит письмо с подтверждением",
      "expected_keywords": ["спам", "проверьте", "папка", "email"],
      "category": "technical"
    }
  ]
}
```

### Скрипт для тестирования

Создайте файл `scripts/evaluate_model.py`:

```python
import json
import requests
import re

def test_model(question, expected_keywords):
    """Тестирует модель на конкретном вопросе"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": "support-bot",
        "prompt": f"Вопрос клиента: {question}\nОтвет:",
        "stream": False
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    answer = result["response"]
    
    # Проверяем наличие ключевых слов
    found_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            found_keywords.append(keyword)
    
    score = len(found_keywords) / len(expected_keywords)
    
    return {
        "question": question,
        "answer": answer,
        "found_keywords": found_keywords,
        "score": score
    }

# Загрузка тестовых данных
with open("data/training/test_set.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Запуск тестов
results = []
for test_case in test_data["test_cases"]:
    result = test_model(
        test_case["question"], 
        test_case["expected_keywords"]
    )
    results.append(result)

# Вывод результатов
total_score = sum(r["score"] for r in results) / len(results)
print(f"Общий балл модели: {total_score:.2f}")

for result in results:
    print(f"\nВопрос: {result['question']}")
    print(f"Ответ: {result['answer']}")
    print(f"Найденные ключевые слова: {result['found_keywords']}")
    print(f"Балл: {result['score']:.2f}")
```

## 🔧 Автоматизация дообучения

### Скрипт для автоматического дообучения

Создайте файл `scripts/auto_retrain.py`:

```python
import json
import os
import subprocess
from datetime import datetime

def collect_new_data():
    """Собирает новые данные из логов"""
    log_file = "data/logs/chat_log.json"
    
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
    
    # Фильтруем качественные диалоги
    quality_conversations = []
    for log in logs:
        if len(log["answer"]) > 50 and "ошибка" not in log["answer"].lower():
            quality_conversations.append({
                "question": log["question"],
                "answer": log["answer"],
                "timestamp": log["timestamp"]
            })
    
    return quality_conversations

def update_training_data(new_conversations):
    """Обновляет данные для обучения"""
    training_file = "data/training/support_conversations.json"
    
    if os.path.exists(training_file):
        with open(training_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"conversations": []}
    
    # Добавляем новые диалоги
    for conv in new_conversations:
        data["conversations"].append(conv)
    
    # Сохраняем обновленные данные
    with open(training_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return len(new_conversations)

def retrain_model():
    """Запускает дообучение модели"""
    print("Начинаем дообучение модели...")
    
    # Создаем новую модель
    subprocess.run(["ollama", "create", "support-bot-v2", "-f", "Modelfile"])
    
    print("Дообучение завершено!")
    return True

def main():
    """Основная функция автоматического дообучения"""
    print(f"Запуск автоматического дообучения: {datetime.now()}")
    
    # Собираем новые данные
    new_data = collect_new_data()
    print(f"Найдено {len(new_data)} новых качественных диалогов")
    
    if len(new_data) >= 10:  # Дообучаем только если есть достаточно данных
        # Обновляем данные для обучения
        updated_count = update_training_data(new_data)
        print(f"Добавлено {updated_count} новых диалогов в данные для обучения")
        
        # Запускаем дообучение
        retrain_model()
        
        print("Автоматическое дообучение завершено успешно!")
    else:
        print("Недостаточно данных для дообучения (минимум 10 диалогов)")

if __name__ == "__main__":
    main()
```

## 📅 Планировщик дообучения

### Настройка cron (Linux/Mac)

```bash
# Добавьте в crontab для ежедневного дообучения в 2:00
0 2 * * * cd /path/to/rag-support-bot && python scripts/auto_retrain.py
```

### Настройка Task Scheduler (Windows)

1. Откройте "Планировщик заданий"
2. Создайте новое задание
3. Установите ежедневное выполнение в 2:00
4. Команда: `python scripts/auto_retrain.py`

## 🎯 Рекомендации по дообучению

1. **Качество данных важнее количества** - лучше 100 качественных диалогов, чем 1000 плохих
2. **Регулярное обновление** - дообучайте модель каждую неделю
3. **Мониторинг качества** - отслеживайте метрики качества ответов
4. **A/B тестирование** - сравнивайте старую и новую модели
5. **Резервные копии** - сохраняйте рабочие версии моделей

---
**Теперь у вас есть полноценная система дообучения для RAG Support Bot! 🎉**
