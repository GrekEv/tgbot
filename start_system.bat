@echo off
echo ========================================
echo    Запуск RAG Support Bot системы
echo ========================================
echo.

echo Шаг 1: Запуск Ollama сервера...
start "Ollama Server" cmd /k "ollama serve"
timeout /t 3 /nobreak > nul

echo Шаг 2: Запуск CORS прокси...
start "CORS Proxy" cmd /k "python cors_proxy.py"
timeout /t 3 /nobreak > nul

echo Шаг 3: Открытие веб-панели...
start support_dashboard.html

echo.
echo ========================================
echo           Система запущена!
echo ========================================
echo.
echo Что запущено:
echo - Ollama сервер: http://localhost:11434
echo - CORS прокси: http://localhost:8080
echo - Веб-панель: support_dashboard.html
echo.
echo Для остановки закройте окна терминалов
echo.
pause
