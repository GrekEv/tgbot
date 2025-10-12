@echo off
echo ========================================
echo    Загрузка RAG Support Bot в GitHub
echo ========================================
echo.

echo Шаг 1: Инициализация Git репозитория...
git init

echo.
echo Шаг 2: Добавление файлов...
git add .

echo.
echo Шаг 3: Создание коммита...
git commit -m "Initial commit - RAG Support Bot for cloud deployment"

echo.
echo Шаг 4: Настройка основной ветки...
git branch -M main

echo.
echo ========================================
echo           ВАЖНО!
echo ========================================
echo.
echo Теперь нужно:
echo 1. Создать репозиторий на GitHub.com
echo 2. Скопировать URL репозитория
echo 3. Выполнить команды:
echo.
echo git remote add origin https://github.com/ВАШ-USERNAME/rag-support-bot.git
echo git push -u origin main
echo.
echo После этого можно развертывать на Render.com
echo.
pause
