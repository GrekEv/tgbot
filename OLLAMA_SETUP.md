# 🦙 Настройка Ollama для локального RAG Support Bot

## 🎯 Цель
Настроить локальную Ollama для работы с RAG Support Bot и возможности дообучения моделей.

## 📋 Что нужно

1. **Windows 10/11** с минимум 8GB RAM
2. **Python 3.8+**
3. **Git** (для клонирования Ollama)
4. **CUDA** (опционально, для GPU ускорения)

## 🚀 Пошаговая установка

### Шаг 1: Установка Ollama

**Для Windows:**
```bash
# Скачайте установщик с https://ollama.com/download
# Или через PowerShell:
winget install Ollama.Ollama
```

**Для Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Шаг 2: Запуск Ollama сервера

```bash
# Запустите Ollama в фоновом режиме
ollama serve
```

Ollama будет доступен по адресу: `http://localhost:11434`

### Шаг 3: Установка моделей

**Рекомендуемые модели для поддержки:**

```bash
# Небольшая быстрая модель (2GB)
ollama pull llama3.2:1b

# Средняя модель (4GB) - рекомендуется
ollama pull llama3.2:3b

# Большая модель (8GB) - лучшее качество
ollama pull llama3.2:8b

# Специальная модель для русского языка
ollama pull saiga3:8b

# Модель для дообучения
ollama pull llama3.2:8b
```

### Шаг 4: Настройка переменных окружения

Создайте файл `.env`:

```env
# Настройки для Ollama
USE_DEEPSEEK=false
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=llama3.2:8b

# Порт приложения
PORT=7860

# Настройки для дообучения
TRAINING_DATA_PATH=./data/training/
MODEL_OUTPUT_PATH=./models/
```

### Шаг 5: Модификация app.py для Ollama

Создайте файл `app_ollama.py`:

```python
import os
import json
import requests
import gradio as gr
from datetime import datetime

# Настройки Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:8b")

def call_ollama_api(question, context=""):
    """Вызывает локальную Ollama API"""
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Формируем промпт
        if context:
            prompt = f"""Ты оператор службы поддержки. На основе контекста ответь на вопрос клиента.

Контекст: {context}

Вопрос клиента: {question}

Ответь профессионально и вежливо на русском языке:"""
        else:
            prompt = f"Ты оператор службы поддержки. Ответь на вопрос клиента: {question}"
        
        data = {
            "model": MODEL_NAME,
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
        return f"Ошибка при обращении к Ollama: {str(e)}"

# Остальной код аналогичен app.py...
```

## 🎓 Дообучение моделей

### Подготовка данных для дообучения

1. **Создайте папку для данных:**
```bash
mkdir -p data/training
```

2. **Создайте файл с примерами диалогов:**
```json
{
  "conversations": [
    {
      "question": "Как сбросить пароль?",
      "answer": "Для сброса пароля перейдите на страницу входа и нажмите 'Забыли пароль?'. Введите ваш email и следуйте инструкциям в письме."
    },
    {
      "question": "Не работает оплата",
      "answer": "Проверьте данные карты и попробуйте другой способ оплаты. Если проблема остается, обратитесь в банк."
    }
  ]
}
```

### Создание кастомной модели

1. **Создайте Modelfile:**
```dockerfile
FROM llama3.2:8b

# Системный промпт
SYSTEM """Ты профессиональный оператор службы поддержки. 
Твоя задача - помогать клиентам решать их проблемы вежливо и эффективно.
Используй только информацию из контекста для ответов."""

# Параметры модели
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 500
```

2. **Создайте модель:**
```bash
ollama create support-bot -f Modelfile
```

3. **Запустите кастомную модель:**
```bash
ollama run support-bot
```

### Дообучение с помощью LoRA

1. **Установите инструменты для дообучения:**
```bash
pip install transformers datasets accelerate peft
```

2. **Создайте скрипт дообучения:**
```python
# train_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

# Загрузите модель
model_name = "meta-llama/Llama-3.2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Настройка LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Дообучение...
```

## 🔧 Интеграция с панелью поддержки

### Модификация support_dashboard.html

Добавьте в JavaScript функцию для работы с Ollama:

```javascript
async function callOllamaAPI(question, context) {
    try {
        const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'support-bot',
                prompt: `Контекст: ${context}\n\nВопрос: ${question}\n\nОтвет:`,
                stream: false,
                options: {
                    temperature: 0.7,
                    max_tokens: 500
                }
            })
        });
        
        const data = await response.json();
        return data.response;
    } catch (error) {
        return `Ошибка: ${error.message}`;
    }
}
```

## 📊 Мониторинг и оптимизация

### Проверка работы Ollama

```bash
# Проверка статуса
curl http://localhost:11434/api/tags

# Тест модели
ollama run llama3.2:8b "Привет, как дела?"

# Мониторинг ресурсов
ollama ps
```

### Оптимизация производительности

1. **Использование GPU:**
```bash
# Установите CUDA для Windows
# Ollama автоматически использует GPU если доступен
```

2. **Настройка памяти:**
```bash
# В .env файле
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=1
```

## 🚀 Запуск полной системы

1. **Запустите Ollama:**
```bash
ollama serve
```

2. **Запустите приложение:**
```bash
python app_ollama.py
```

3. **Откройте панель поддержки:**
```bash
start support_dashboard.html
```

## 🎯 Преимущества локальной Ollama

- ✅ **Полная приватность** - данные не покидают ваш компьютер
- ✅ **Бесплатное использование** - нет лимитов API
- ✅ **Дообучение моделей** - можно адаптировать под ваши задачи
- ✅ **Офлайн работа** - не нужен интернет
- ✅ **Быстрые ответы** - нет задержек сети

## 🔧 Решение проблем

**Проблема:** Ollama не запускается
```bash
# Перезапустите службу
ollama serve
```

**Проблема:** Модель не загружается
```bash
# Проверьте доступную память
ollama ps
# Удалите неиспользуемые модели
ollama rm model_name
```

**Проблема:** Медленные ответы
```bash
# Используйте меньшую модель
ollama pull llama3.2:3b
```

---
**Готово! Теперь у вас есть полноценная локальная система поддержки с возможностью дообучения! 🎉**
