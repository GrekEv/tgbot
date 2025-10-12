FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаём папку для данных
RUN mkdir -p data/logs

# Открываем порт
EXPOSE 7860

# Устанавливаем переменные окружения
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Запускаем приложение
CMD ["python", "app.py"]
