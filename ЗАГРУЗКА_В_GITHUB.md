# 📤 Загрузка проекта в GitHub (без Git)

## 🎯 Пошаговая инструкция:

### 1️⃣ Создание репозитория на GitHub

**Зайдите на GitHub:**
1. Откройте https://github.com
2. Войдите в свой аккаунт
3. Нажмите зеленую кнопку **"New"** или **"+"** → **"New repository"**

**Настройки репозитория:**
- **Repository name:** `rag-support-bot`
- **Description:** `RAG Support Bot - Система поддержки банка с AI`
- **Visibility:** ✅ **Public** (для бесплатного Render)
- **Initialize:** ❌ НЕ ставьте галочки (README, .gitignore, license)
- Нажмите **"Create repository"**

### 2️⃣ Загрузка файлов через веб-интерфейс

**На странице репозитория:**
1. Нажмите **"uploading an existing file"**
2. Перетащите файлы из папки проекта или нажмите **"choose your files"**

**Загрузите эти файлы:**
- ✅ `streamlit_app.py`
- ✅ `requirements.txt`
- ✅ `render.yaml`
- ✅ `railway.json`
- ✅ `support_dashboard.html`
- ✅ `app_simple_ollama.py`
- ✅ `cors_proxy.py`
- ✅ `BankModelfile`
- ✅ `Modelfile`
- ✅ `bank_support_prompt.txt`
- ✅ Все `.md` файлы (документация)

**После загрузки:**
1. Внизу страницы введите:
   - **Commit message:** `Initial commit - RAG Support Bot`
2. Нажмите **"Commit changes"**

### 3️⃣ Развертывание на Render.com

**Создайте аккаунт на Render:**
1. Зайдите на https://render.com
2. Нажмите **"Get Started for Free"**
3. Войдите через **GitHub** аккаунт

**Создайте Web Service:**
1. Нажмите **"New +"** → **"Web Service"**
2. Подключите GitHub репозиторий:
   - Нажмите **"Connect account"** если нужно
   - Выберите репозиторий **`rag-support-bot`**
   - Нажмите **"Connect"**

**Настройки сервиса:**
- **Name:** `rag-support-bot`
- **Environment:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- **Plan:** `Free`

**Переменные окружения:**
1. В разделе **"Environment Variables"** добавьте:
   - **Key:** `PORT` **Value:** `10000`
   - **Key:** `USE_OLLAMA` **Value:** `false`
   - **Key:** `OPENAI_API_KEY` **Value:** (оставьте пустым, пользователи будут вводить свой)

**Запуск:**
1. Нажмите **"Create Web Service"**
2. Дождитесь сборки (5-10 минут)
3. Получите ссылку: `https://rag-support-bot.onrender.com`

## 🎉 Готово!

### ✅ Что получите:
- **Публичная ссылка:** `https://rag-support-bot.onrender.com`
- **Веб-интерфейс** банковской поддержки
- **AI подсказки** для операторов
- **Доступ 24/7** для всей команды

### 🚀 Как поделиться с командой:
1. Отправьте ссылку: `https://rag-support-bot.onrender.com`
2. Каждый открывает в браузере
3. Вводит свой OpenAI API ключ (опционально)
4. Начинает работать с системой

## 🔧 Альтернативные варианты:

### Если Render не работает:

**Railway.app:**
1. Зайдите на https://railway.app
2. Нажмите **"Deploy from GitHub repo"**
3. Выберите `rag-support-bot`
4. Получите ссылку: `https://rag-support-bot-production.up.railway.app`

**Streamlit Cloud:**
1. Зайдите на https://share.streamlit.io
2. Нажмите **"Deploy an app"**
3. Выберите репозиторий `rag-support-bot`
4. Файл: `streamlit_app.py`
5. Получите ссылку: `https://rag-support-bot.streamlit.app`

## 📱 Что увидит команда:

### Веб-интерфейс включает:
- 💬 **Список чатов** с клиентами
- 💭 **Диалоги** в реальном времени  
- 🤖 **AI подсказки** для операторов
- ⚙️ **Настройки** API и оператора
- 📊 **Статистика** использования

### AI функции:
- 🔍 **Классификация** запросов клиентов
- 💡 **Предложение решений** проблем
- 📞 **Передача специалисту** сложных вопросов
- ❓ **Запрос уточнений** для лучшего понимания

---
**Следуйте инструкции и получите рабочую ссылку для команды! 🚀**
