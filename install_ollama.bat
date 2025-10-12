@echo off
echo ========================================
echo    Установка Ollama для RAG Support Bot
echo ========================================
echo.

echo Шаг 1: Скачивание и установка Ollama...
echo.

REM Скачиваем установщик Ollama для Windows
echo Скачиваем Ollama...
powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/windows' -OutFile 'ollama-windows-amd64.exe'"

echo.
echo Запускаем установщик...
ollama-windows-amd64.exe

echo.
echo Шаг 2: Запуск Ollama сервера...
echo.

REM Запускаем Ollama в фоновом режиме
start /B ollama serve

echo Ожидаем запуска сервера...
timeout /t 5 /nobreak > nul

echo.
echo Шаг 3: Установка модели...
echo.

REM Скачиваем рекомендуемую модель
echo Скачиваем модель llama3.2:8b (это может занять несколько минут)...
ollama pull llama3.2:8b

echo.
echo Шаг 4: Создание .env файла...
echo.

REM Создаем .env файл для Ollama
echo USE_OLLAMA=true > .env
echo OLLAMA_HOST=http://localhost:11434 >> .env
echo MODEL_NAME=llama3.2:8b >> .env
echo PORT=7860 >> .env
echo OPENAI_API_KEY=your-openai-key-for-embeddings >> .env

echo.
echo ========================================
echo           Установка завершена!
echo ========================================
echo.
echo Что дальше:
echo 1. Получите OpenAI API ключ для embeddings на https://platform.openai.com/
echo 2. Отредактируйте .env файл и добавьте ваш ключ
echo 3. Запустите: python app_ollama.py
echo 4. Откройте: http://localhost:7860
echo.
echo Для дообучения моделей смотрите OLLAMA_SETUP.md
echo.
pause
