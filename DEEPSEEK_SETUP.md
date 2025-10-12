# 🔵 Настройка DeepSeek API

## Шаги для получения бесплатного API ключа DeepSeek:

### 1. Регистрация
1. Перейдите на https://platform.deepseek.com/
2. Нажмите "Sign Up" или "Регистрация"
3. Заполните форму регистрации (email, пароль)
4. Подтвердите email

### 2. Получение API ключа
1. Войдите в личный кабинет
2. Перейдите в раздел "API Keys" или "API ключи"
3. Нажмите "Create New Key" или "Создать новый ключ"
4. Скопируйте полученный API ключ

### 3. Настройка проекта
Создайте файл `.env` в корне проекта со следующим содержимым:

```env
# Настройки для DeepSeek API
USE_DEEPSEEK=true
DEEPSEEK_API_KEY=ваш-api-ключ-здесь

# Общие настройки
PORT=7860
```

### 4. Установка зависимостей
```bash
pip install gradio langchain faiss-cpu openai pandas numpy tiktoken python-dotenv requests
```

### 5. Запуск
```bash
python app.py
```

## Преимущества DeepSeek API:
- ✅ Бесплатный доступ
- ✅ Высокое качество ответов на русском языке
- ✅ Быстрая скорость обработки
- ✅ Совместимость с OpenAI API

## Альтернативные варианты:
Если у вас есть проблемы с DeepSeek, можете использовать:
- OpenAI API (платный)
- Локальную Ollama с моделью mistral
- Другие бесплатные API

## Поддержка:
- Официальная документация: https://platform.deepseek.com/docs
- Telegram бот для тестирования: @DeepseekR1FreeApi
