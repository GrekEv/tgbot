import os
import json
import requests
import gradio as gr
from datetime import datetime

# Настройки для DeepSeek API
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Загружаем API ключ из переменных окружения
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Простая база знаний FAQ
FAQ_DATA = [
    {"question": "Как сбросить пароль?", "answer": "Перейдите на страницу входа и нажмите 'Забыли пароль?'. Введите ваш email и следуйте инструкциям в письме."},
    {"question": "Как изменить email?", "answer": "Войдите в настройки профиля, найдите раздел 'Контактная информация' и обновите email адрес."},
    {"question": "Как связаться с поддержкой?", "answer": "Вы можете связаться с нами через email support@example.com или через форму обратной связи на сайте."},
    {"question": "Какие способы оплаты доступны?", "answer": "Мы принимаем банковские карты, PayPal, Apple Pay и Google Pay."},
    {"question": "Как отменить заказ?", "answer": "Заказы можно отменить в течение 24 часов после оформления через личный кабинет или обратившись в поддержку."}
]

def call_deepseek_api(question, context=""):
    """Вызывает DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "❌ API ключ DeepSeek не установлен. Получите бесплатный ключ на https://platform.deepseek.com/"
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Формируем промпт с контекстом
        if context:
            prompt = f"""На основе следующей информации ответь на вопрос пользователя:

Контекст: {context}

Вопрос: {question}

Ответь кратко и по существу на русском языке:"""
        else:
            prompt = f"Ответь на вопрос: {question}"
        
        data = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"❌ Ошибка при обращении к DeepSeek API: {str(e)}"

def find_relevant_faq(question):
    """Находит релевантные FAQ (простой поиск по ключевым словам)"""
    question_lower = question.lower()
    relevant_faqs = []
    
    for faq in FAQ_DATA:
        # Простой поиск по ключевым словам
        if any(word in question_lower for word in faq["question"].lower().split()):
            relevant_faqs.append(faq)
    
    return relevant_faqs

def get_response(question):
    """Генерирует ответ на вопрос"""
    if not question.strip():
        return "Пожалуйста, задайте вопрос."
    
    # Ищем релевантные FAQ
    relevant_faqs = find_relevant_faq(question)
    
    if relevant_faqs:
        # Если нашли релевантные FAQ, используем их как контекст
        context = "\n".join([f"В: {faq['question']}\nО: {faq['answer']}" for faq in relevant_faqs])
        answer = call_deepseek_api(question, context)
    else:
        # Если не нашли, отвечаем напрямую
        answer = call_deepseek_api(question)
    
    # Логируем взаимодействие
    log_interaction(question, answer)
    
    return answer

def log_interaction(question, answer):
    """Логирует взаимодействие"""
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    }
    
    # Сохраняем в простой текстовый файл
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{log['timestamp']} | {question} | {answer}\n")

def chat_fn(message, history):
    """Функция для чата"""
    if not message.strip():
        return history, ""
    
    answer = get_response(message)
    history.append((message, answer))
    return history, ""

# Создаем интерфейс
with gr.Blocks(title="RAG Support Bot - DeepSeek", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Support Bot с DeepSeek API")
    
    with gr.Tab("💬 Чат"):
        gr.Markdown("Задайте вопрос боту поддержки. Он найдет ответ в базе знаний.")
        chatbot = gr.Chatbot(label="Диалог с ботом", height=400)
        msg = gr.Textbox(label="Введите вопрос", placeholder="Как сбросить пароль?")
        send = gr.Button("Отправить", variant="primary")
        send.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

    with gr.Tab("📚 FAQ база"):
        gr.Markdown("Текущие вопросы в базе знаний:")
        faq_text = ""
        for i, faq in enumerate(FAQ_DATA, 1):
            faq_text += f"{i}. **В:** {faq['question']}\n   **О:** {faq['answer']}\n\n"
        gr.Markdown(faq_text)

    with gr.Tab("⚙️ Настройки"):
        gr.Markdown("### Текущие настройки")
        gr.Markdown(f"**🔵 API Провайдер:** DeepSeek")
        gr.Markdown(f"**Модель:** {DEEPSEEK_MODEL}")
        gr.Markdown(f"**API Key:** {'✅ Установлен' if DEEPSEEK_API_KEY else '❌ Не установлен'}")
        
        gr.Markdown("### Инструкции по настройке")
        gr.Markdown("""
        1. **Получите API ключ DeepSeek:**
           - Зарегистрируйтесь на https://platform.deepseek.com/
           - Создайте API ключ в личном кабинете
           
        2. **Установите переменную окружения:**
           ```bash
           set DEEPSEEK_API_KEY=ваш-ключ-здесь
           ```
           
        3. **Или создайте файл .env:**
           ```
           DEEPSEEK_API_KEY=ваш-ключ-здесь
           ```
        """)

if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("⚠️  ВНИМАНИЕ: DEEPSEEK_API_KEY не установлен!")
        print("Получите бесплатный API ключ на https://platform.deepseek.com/")
        print("Установите переменную окружения: set DEEPSEEK_API_KEY=your-key")
    else:
        print("✅ DeepSeek API ключ найден")
    
    port = int(os.getenv("PORT", 7860))
    print(f"🚀 Запуск RAG Support Bot на порту {port}")
    print(f"🌐 Откройте браузер: http://localhost:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
