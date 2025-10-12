import streamlit as st
import requests
import json
import os
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="RAG Support Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .support-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .ai-suggestion {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
    }
    .ai-suggestion:hover {
        background-color: #ffe0b2;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🏦 RAG Support Bot - Банковская поддержка</h1>', unsafe_allow_html=True)

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор API
    api_provider = st.selectbox(
        "Провайдер API",
        ["OpenAI", "DeepSeek", "Локальная Ollama"]
    )
    
    if api_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="Получите ключ на https://platform.openai.com")
        model_name = st.selectbox("Модель", ["gpt-3.5-turbo", "gpt-4"])
    elif api_provider == "DeepSeek":
        api_key = st.text_input("DeepSeek API Key", type="password", help="Получите ключ на https://platform.deepseek.com")
        model_name = "deepseek-chat"
    else:
        api_key = ""
        model_name = "support-bot"
        st.info("Используется локальная Ollama модель")
    
    # Настройки оператора
    st.header("👤 Оператор")
    operator_name = st.text_input("Имя оператора", value="Оператор поддержки")
    company_name = st.text_input("Название компании", value="Наш банк")
    
    # Статистика
    st.header("📊 Статистика")
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    if 'ai_suggestions_used' not in st.session_state:
        st.session_state.ai_suggestions_used = 0
    
    st.metric("Всего сообщений", st.session_state.total_messages)
    st.metric("AI подсказок использовано", st.session_state.ai_suggestions_used)

# Основной интерфейс
col1, col2 = st.columns([1, 2])

with col1:
    st.header("💬 Чаты с клиентами")
    
    # Список чатов
    if 'chats' not in st.session_state:
        st.session_state.chats = {
            "Клиент 1": {
                "messages": [
                    {"type": "user", "text": "Здравствуйте! Не могу войти в мобильный банк", "time": "10:30"},
                    {"type": "support", "text": "Здравствуйте! Помогу вам решить проблему с входом", "time": "10:31"}
                ],
                "status": "active"
            },
            "Клиент 2": {
                "messages": [
                    {"type": "user", "text": "Какие комиссии за переводы?", "time": "11:15"}
                ],
                "status": "waiting"
            },
            "Клиент 3": {
                "messages": [
                    {"type": "user", "text": "Хочу сменить номер телефона", "time": "11:45"}
                ],
                "status": "waiting"
            }
        }
    
    # Отображение чатов
    for chat_name, chat_data in st.session_state.chats.items():
        status_emoji = "🟢" if chat_data["status"] == "active" else "🟡"
        last_message = chat_data["messages"][-1]["text"][:50] + "..." if len(chat_data["messages"][-1]["text"]) > 50 else chat_data["messages"][-1]["text"]
        
        if st.button(f"{status_emoji} {chat_name}\n{last_message}", key=f"chat_{chat_name}"):
            st.session_state.current_chat = chat_name

with col2:
    st.header("💭 Диалог с клиентом")
    
    # Выбранный чат
    if 'current_chat' in st.session_state:
        current_chat = st.session_state.current_chat
        st.subheader(f"Чат: {current_chat}")
        
        # Отображение сообщений
        chat_messages = st.session_state.chats[current_chat]["messages"]
        
        for message in chat_messages:
            if message["type"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>Клиент:</strong> {message["text"]}<br><small>{message["time"]}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message support-message"><strong>{operator_name}:</strong> {message["text"]}<br><small>{message["time"]}</small></div>', unsafe_allow_html=True)
        
        # Поле ввода
        st.subheader("✍️ Ответ клиенту")
        user_input = st.text_area("Введите ваш ответ:", height=100, key="message_input")
        
        # Кнопки отправки
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            if st.button("📤 Отправить", type="primary"):
                if user_input:
                    # Добавляем сообщение оператора
                    new_message = {
                        "type": "support",
                        "text": user_input,
                        "time": datetime.now().strftime("%H:%M")
                    }
                    st.session_state.chats[current_chat]["messages"].append(new_message)
                    st.session_state.total_messages += 1
                    st.rerun()
        
        with col2_2:
            if st.button("🔄 Обновить"):
                st.rerun()
    else:
        st.info("Выберите чат для начала работы")

# AI подсказки
st.header("🤖 AI Подсказки")

if 'current_chat' in st.session_state and st.session_state.chats[st.session_state.current_chat]["messages"]:
    # Получаем последнее сообщение клиента
    last_user_message = None
    for message in reversed(st.session_state.chats[st.session_state.current_chat]["messages"]):
        if message["type"] == "user":
            last_user_message = message["text"]
            break
    
    if last_user_message:
        st.info(f"Анализируем: '{last_user_message}'")
        
        # AI подсказки
        col3, col4, col5 = st.columns(3)
        
        with col3:
            if st.button("🔍 Классифицировать запрос"):
                if api_key or api_provider == "Локальная Ollama":
                    # Здесь будет вызов AI API
                    st.session_state.ai_suggestions_used += 1
                    st.success("Запрос классифицирован как: 'Техническая проблема'")
                    st.info("Рекомендация: Уточните, с чем именно возникли сложности при входе в мобильный банк")
                else:
                    st.error("Укажите API ключ в настройках")
        
        with col4:
            if st.button("💡 Предложить решение"):
                if api_key or api_provider == "Локальная Ollama":
                    st.session_state.ai_suggestions_used += 1
                    st.success("Решение предложено!")
                    st.info("Рекомендация: Проверьте правильность ввода логина и пароля, попробуйте сбросить пароль через 'Забыли пароль?'")
                else:
                    st.error("Укажите API ключ в настройках")
        
        with col5:
            if st.button("📞 Передать специалисту"):
                if api_key or api_provider == "Локальная Ollama":
                    st.session_state.ai_suggestions_used += 1
                    st.success("Запрос передан специалисту!")
                    st.info("Рекомендация: Ваш вопрос будет передан техническому специалисту. Ожидайте ответа в течение 2-4 часов в рабочие дни.")
                else:
                    st.error("Укажите API ключ в настройках")
        
        # Дополнительные подсказки
        col6, col7 = st.columns(2)
        
        with col6:
            if st.button("❓ Запросить уточнения"):
                if api_key or api_provider == "Локальная Ollama":
                    st.session_state.ai_suggestions_used += 1
                    st.success("Уточнения сформулированы!")
                    st.info("Рекомендация: Пожалуйста, уточните: когда возникла проблема, какое сообщение об ошибке появляется, пробовали ли вы перезапустить приложение?")
                else:
                    st.error("Укажите API ключ в настройках")
        
        with col7:
            if st.button("📊 Анализ проблемы"):
                if api_key or api_provider == "Локальная Ollama":
                    st.session_state.ai_suggestions_used += 1
                    st.success("Анализ выполнен!")
                    st.info("Рекомендация: Проблема связана с авторизацией в мобильном приложении. Необходимо проверить учетные данные и статус аккаунта.")
                else:
                    st.error("Укажите API ключ в настройках")
    else:
        st.warning("Нет сообщений от клиента для анализа")
else:
    st.info("Выберите чат с сообщениями клиента для использования AI подсказок")

# Подвал
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏦 RAG Support Bot - Система поддержки банка с AI</p>
    <p>Версия 2.0 | Создано с помощью Streamlit</p>
</div>
""", unsafe_allow_html=True)
