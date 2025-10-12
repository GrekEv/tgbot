@echo off
echo ========================================
echo    Остановка RAG Support Bot системы
echo ========================================
echo.

echo Остановка Ollama сервера...
taskkill /F /IM ollama.exe 2>nul
if %errorlevel% == 0 (
    echo ✅ Ollama сервер остановлен
) else (
    echo ⚠️  Ollama сервер не был запущен
)

echo.
echo Остановка CORS прокси...
taskkill /F /IM python.exe 2>nul
if %errorlevel% == 0 (
    echo ✅ CORS прокси остановлен
) else (
    echo ⚠️  CORS прокси не был запущен
)

echo.
echo Остановка веб-панели...
taskkill /F /IM chrome.exe 2>nul
taskkill /F /IM firefox.exe 2>nul
taskkill /F /IM msedge.exe 2>nul

echo.
echo ========================================
echo         Система остановлена!
echo ========================================
echo.
pause
