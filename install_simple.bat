@echo off
echo Установка минимальных зависимостей для RAG Support Bot...
echo.

echo Установка Gradio...
pip install gradio

echo.
echo Установка requests...
pip install requests

echo.
echo Установка python-dotenv...
pip install python-dotenv

echo.
echo Установка завершена!
echo.
echo Следующие шаги:
echo 1. Получите API ключ DeepSeek на https://platform.deepseek.com/
echo 2. Установите переменную окружения: set DEEPSEEK_API_KEY=ваш-ключ
echo 3. Запустите: python simple_app.py
echo.
pause
