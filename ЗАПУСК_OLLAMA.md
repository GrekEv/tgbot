# 🚀 Инструкция по запуску Ollama сервера

## 📋 Что нужно для запуска:

### ✅ Требования:
- **Ollama установлен** (версия 0.12.5+)
- **Python 3.8+** (для CORS прокси)
- **Минимум 4GB RAM** (рекомендуется 8GB)
- **Свободный порт 11434** (Ollama)
- **Свободный порт 8080** (CORS прокси)

## 🔧 Пошаговая инструкция:

### 1️⃣ Проверка установки Ollama

**Проверьте, что Ollama установлен:**
```bash
ollama --version
```

**Если команда не найдена:**
- Скачайте Ollama с https://ollama.com/download
- Установите и перезагрузите компьютер
- Добавьте Ollama в PATH

### 2️⃣ Запуск Ollama сервера

**Способ 1: Через командную строку**
```bash
# Откройте PowerShell или CMD
# Перейдите в папку проекта
cd C:\rag-support-bot

# Запустите Ollama сервер
ollama serve
```

**Способ 2: Через полный путь**
```bash
# Если ollama не в PATH
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
```

**Способ 3: Через batch файл**
```bash
# Создайте файл start_ollama.bat
echo @echo off > start_ollama.bat
echo echo Запуск Ollama сервера... >> start_ollama.bat
echo ollama serve >> start_ollama.bat
echo pause >> start_ollama.bat

# Запустите
start_ollama.bat
```

### 3️⃣ Проверка работы сервера

**Проверьте статус:**
```bash
# Проверка через curl
curl http://localhost:11434/api/tags

# Проверка через PowerShell
Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET
```

**Ожидаемый ответ:**
```json
{"models": [{"name": "support-bot:latest", "model": "support-bot:latest", ...}]}
```

### 4️⃣ Запуск CORS прокси

**В новом окне терминала:**
```bash
# Перейдите в папку проекта
cd C:\rag-support-bot

# Запустите CORS прокси
python cors_proxy.py
```

**Ожидаемый вывод:**
```
🚀 Запуск CORS прокси на порту 8080
🌐 Проксирует запросы к Ollama (localhost:11434)
📡 Доступен по адресу: http://localhost:8080
==================================================
✅ Сервер запущен на http://localhost:8080
🔄 Проксирует запросы к Ollama...
⏹️  Для остановки нажмите Ctrl+C
```

### 5️⃣ Проверка CORS прокси

**Проверьте прокси:**
```bash
# Проверка через curl
curl http://localhost:8080/api/tags

# Проверка через PowerShell
Invoke-WebRequest -Uri "http://localhost:8080/api/tags" -Method GET
```

## 🎯 Полный запуск системы:

### 📝 Создайте batch файл для автоматического запуска:

**Файл: `start_system.bat`**
```batch
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
```

### 🚀 Запуск через batch файл:

```bash
# Создайте файл start_system.bat
# Запустите его
start_system.bat
```

## 🔧 Решение проблем:

### ❌ Ошибка: "Only one usage of each socket address"

**Причина:** Порт 11434 уже занят

**Решение:**
```bash
# Найдите процесс, использующий порт
netstat -ano | findstr :11434

# Завершите процесс (замените PID на реальный)
taskkill /PID <PID> /F

# Запустите Ollama заново
ollama serve
```

### ❌ Ошибка: "ollama: command not found"

**Причина:** Ollama не в PATH

**Решение:**
```bash
# Используйте полный путь
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve

# Или добавьте в PATH
$env:PATH += ";C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama"
```

### ❌ Ошибка: "Failed to fetch"

**Причина:** CORS прокси не запущен

**Решение:**
```bash
# Запустите CORS прокси
python cors_proxy.py

# Проверьте, что он работает
curl http://localhost:8080/api/tags
```

### ❌ Ошибка: "Python not found"

**Причина:** Python не установлен

**Решение:**
```bash
# Установите Python с https://python.org
# Или используйте встроенный Python
py cors_proxy.py
```

## 📊 Мониторинг системы:

### 🔍 Проверка статуса:

**Ollama сервер:**
```bash
# Статус сервера
curl http://localhost:11434/api/tags

# Список моделей
ollama list

# Использование ресурсов
ollama ps
```

**CORS прокси:**
```bash
# Статус прокси
curl http://localhost:8080/api/tags

# Проверка порта
netstat -an | findstr :8080
```

### 📈 Логи и отладка:

**Ollama логи:**
- Логи выводятся в консоль сервера
- Для отладки запустите: `ollama serve --verbose`

**CORS прокси логи:**
- Логи выводятся в консоль прокси
- Ошибки показываются в реальном времени

## 🎯 Автоматизация:

### 📅 Запуск при старте системы:

**Создайте задачу в Планировщике заданий:**
1. Откройте "Планировщик заданий"
2. Создайте новое задание
3. Установите триггер "При запуске системы"
4. Действие: запуск `start_system.bat`

### 🔄 Автоматический перезапуск:

**Файл: `restart_system.bat`**
```batch
@echo off
echo Остановка системы...
taskkill /F /IM ollama.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak > nul

echo Запуск системы...
start_system.bat
```

## 🎉 Готово к работе!

### ✅ Что должно работать:

1. **Ollama сервер** - http://localhost:11434
2. **CORS прокси** - http://localhost:8080
3. **Веб-панель** - support_dashboard.html
4. **AI подсказки** - генерируют ответы
5. **Модель support-bot** - готова к использованию

### 🚀 Следующие шаги:

1. **Запустите систему** по инструкции
2. **Откройте веб-панель** support_dashboard.html
3. **Протестируйте AI подсказки**
4. **Добавьте свои FAQ**
5. **Начните использовать систему**

---
**Система готова к работе! 🎉**
