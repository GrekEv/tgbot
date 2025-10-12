@echo off
echo Установка зависимостей для RAG Support Bot...
echo.

echo Установка основных пакетов...
pip install gradio==4.36.1
pip install langchain==0.2.10
pip install faiss-cpu==1.8.0
pip install openai==1.30.1
pip install pandas==2.1.4
pip install numpy==1.24.3
pip install tiktoken==0.5.2
pip install python-dotenv==1.0.0
pip install requests==2.31.0

echo.
echo Установка завершена!
echo.
echo Следующие шаги:
echo 1. Получите API ключ DeepSeek на https://platform.deepseek.com/
echo 2. Создайте файл .env с вашим API ключом
echo 3. Запустите: python app.py
echo.
pause
