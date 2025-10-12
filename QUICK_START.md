# 🚀 Быстрый старт RAG Support Bot

## 📋 Что нужно для запуска

1. **Python 3.11+** установлен на компьютере
2. **OpenAI API ключ** (получить на https://platform.openai.com/api-keys)
3. **Интернет соединение** для работы с OpenAI API

## ⚡ Запуск за 5 минут

### Шаг 1: Установка зависимостей
```bash
# Перейдите в папку проекта
cd rag-support-bot

# Создайте виртуальное окружение
python -m venv venv

# Активируйте его
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

### Шаг 2: Настройка API ключа
```bash
# Создайте .env файл
copy .env.example .env

# Отредактируйте .env файл и добавьте ваш API ключ:
# OPENAI_API_KEY=sk-your-key-here
```

### Шаг 3: Запуск
```bash
python app.py
```

Откройте браузер: http://localhost:7860

## 🎯 Первые шаги

1. **Протестируйте чат**: Задайте вопрос "Как сбросить пароль?"
2. **Добавьте FAQ**: Перейдите на вкладку "FAQ база" и добавьте свой вопрос
3. **Проверьте логи**: Посмотрите историю в разделе "Логи"

## 🔧 Альтернативные способы запуска

### Через run.py (с проверками)
```bash
python run.py
```

### Через Docker
```bash
docker build -t rag-bot .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key rag-bot
```

### Через Docker Compose
```bash
# Создайте .env файл с OPENAI_API_KEY
docker-compose up -d
```

## ❓ Частые проблемы

**Ошибка "No module named 'gradio'"**
→ Запустите: `pip install -r requirements.txt`

**Ошибка "OPENAI_API_KEY not found"**
→ Создайте .env файл с вашим API ключом

**Приложение не запускается**
→ Проверьте, что порт 7860 свободен

**Медленные ответы**
→ Это нормально для первого запуска (создается векторная база)

## 📞 Нужна помощь?

1. Прочитайте полную документацию в README.md
2. Проверьте логи в разделе "Логи" приложения
3. Убедитесь, что API ключ действителен

---
**Готово! Ваш RAG Support Bot работает! 🎉**
