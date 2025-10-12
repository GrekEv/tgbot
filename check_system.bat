@echo off
echo ========================================
echo    Проверка статуса RAG Support Bot
echo ========================================
echo.

echo Проверка Ollama сервера...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama сервер работает (порт 11434)
) else (
    echo ❌ Ollama сервер не отвечает
)

echo.
echo Проверка CORS прокси...
curl -s http://localhost:8080/api/tags >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ CORS прокси работает (порт 8080)
) else (
    echo ❌ CORS прокси не отвечает
)

echo.
echo Проверка портов...
netstat -an | findstr :11434 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Порт 11434 занят (Ollama)
) else (
    echo ❌ Порт 11434 свободен
)

netstat -an | findstr :8080 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Порт 8080 занят (CORS прокси)
) else (
    echo ❌ Порт 8080 свободен
)

echo.
echo Проверка процессов...
tasklist | findstr ollama.exe >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Процесс Ollama запущен
) else (
    echo ❌ Процесс Ollama не найден
)

tasklist | findstr python.exe >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Процесс Python запущен
) else (
    echo ❌ Процесс Python не найден
)

echo.
echo ========================================
echo           Проверка завершена!
echo ========================================
echo.
pause
