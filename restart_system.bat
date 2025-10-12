@echo off
echo ========================================
echo    Перезапуск RAG Support Bot системы
echo ========================================
echo.

echo Остановка системы...
call stop_system.bat

echo.
echo Ожидание 3 секунды...
timeout /t 3 /nobreak > nul

echo.
echo Запуск системы...
call start_system.bat
