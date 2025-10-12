import os
import json
import requests
import gradio as gr
from datetime import datetime

# Настройки Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "support-bot")
PORT = int(os.getenv("PORT", 7860))

# Простая база знаний FAQ (без векторного поиска)
FAQ_DATA = [
    {"question": "Как сбросить пароль?", "answer": "Перейдите на страницу входа и нажмите 'Забыли пароль?'. Введите ваш email и следуйте инструкциям в письме."},
    {"question": "Как изменить email?", "answer": "Войдите в настройки профиля, найдите раздел 'Контактная информация' и обновите email адрес."},
    {"question": "Как связаться с поддержкой?", "answer": "Вы можете связаться с нами через email support@example.com или через форму обратной связи на сайте."},
    {"question": "Какие способы оплаты доступны?", "answer": "Мы принимаем банковские карты, PayPal, Apple Pay и Google Pay."},
    {"question": "Как отменить заказ?", "answer": "Заказы можно отменить в течение 24 часов после оформления через личный кабинет или обратившись в поддержку."},
    {"question": "Не работает оплата", "answer": "Проверьте данные карты и попробуйте другой способ оплаты. Если проблема остается, обратитесь в банк."},
    {"question": "Не приходит письмо с подтверждением", "answer": "Проверьте папку 'Спам' в вашем почтовом ящике. Если письма нет, попробуйте запросить повторную отправку."},
    {"question": "Проблема с загрузкой страницы", "answer": "Попробуйте очистить кэш браузера и обновить страницу. Если проблема остается, проверьте подключение к интернету."}
]

def call_ollama_api(question, context=""):
    """Вызывает локальную Ollama API"""
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Формируем промпт
        if context:
            prompt = f"""Ты профессиональный оператор службы поддержки. На основе контекста ответь на вопрос клиента.

Контекст: {context}

Вопрос клиента: {question}

Ответь профессионально и вежливо на русском языке:"""
        else:
            prompt = f"""Ты профессиональный оператор службы поддержки. Ответь на вопрос клиента: {question}

Ответь вежливо и профессионально на русском языке:"""
        
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
        
    except Exception as e:
        return f"Ошибка при обращении к Ollama: {str(e)}"

def find_relevant_faq(question):
    """Находит релевантные FAQ (простой поиск по ключевым словам)"""
    question_lower = question.lower()
    relevant_faqs = []
    
    for faq in FAQ_DATA:
        # Простой поиск по ключевым словам
        question_words = faq["question"].lower().split()
        if any(word in question_lower for word in question_words if len(word) > 3):
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
        answer = call_ollama_api(question, context)
    else:
        # Если не нашли, отвечаем напрямую
        answer = call_ollama_api(question)
    
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
    
    # Создаем папку логов если нет
    os.makedirs("data/logs", exist_ok=True)
    
    # Загружаем существующие логи
    logs = []
    log_file = "data/logs/chat_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    # Добавляем новый лог
    logs.append(log)
    
    # Сохраняем логи
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def load_logs():
    """Загружает логи чата"""
    log_file = "data/logs/chat_log.json"
    if not os.path.exists(log_file):
        return []
    with open(log_file, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return []

def add_faq(question, answer):
    """Добавляет новый FAQ"""
    if not question.strip() or not answer.strip():
        return "Вопрос и ответ не могут быть пустыми!", FAQ_DATA
    
    new_faq = {"question": question, "answer": answer}
    FAQ_DATA.append(new_faq)
    
    # Сохраняем в файл
    with open("data/faq.json", "w", encoding="utf-8") as f:
        json.dump(FAQ_DATA, f, ensure_ascii=False, indent=2)
    
    return "Добавлено успешно!", FAQ_DATA

def load_faq():
    """Загружает FAQ из файла"""
    if os.path.exists("data/faq.json"):
        with open("data/faq.json", "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return FAQ_DATA
    return FAQ_DATA

def chat_fn(message, history):
    """Основная функция для чата"""
    if not message.strip():
        return history, ""
    
    answer = get_response(message)
    history.append((message, answer))
    return history, ""

def render_logs():
    """Рендерит логи в HTML таблицу"""
    logs = load_logs()
    if not logs:
        return "Логов пока нет."
    
    table = """
    <style>
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
    <table>
    <tr><th>Дата</th><th>Вопрос</th><th>Ответ</th></tr>
    """
    
    for log in reversed(logs[-30:]):  # Показываем последние 30 записей
        timestamp = log['timestamp'][:19].replace('T', ' ')
        question = log['question'][:100] + "..." if len(log['question']) > 100 else log['question']
        answer = log['answer'][:200] + "..." if len(log['answer']) > 200 else log['answer']
        table += f"<tr><td>{timestamp}</td><td>{question}</td><td>{answer}</td></tr>"
    
    table += "</table>"
    return table

def clear_logs():
    """Очищает все логи"""
    log_file = "data/logs/chat_log.json"
    if os.path.exists(log_file):
        os.remove(log_file)
    return "Логи очищены!"

# Создаем интерфейс
with gr.Blocks(title="RAG Support Bot - Ollama", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Support Bot — панель поддержки с Ollama")
    
    with gr.Tab("Чат"):
        gr.Markdown("Задайте вопрос боту поддержки. Он найдет ответ в базе знаний.")
        chatbot = gr.Chatbot(label="Диалог с ботом", height=400)
        msg = gr.Textbox(label="Введите вопрос", placeholder="Как сбросить пароль?")
        send = gr.Button("Отправить", variant="primary")
        send.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

    with gr.Tab("FAQ база"):
        gr.Markdown("Управление базой знаний. Добавляйте новые вопросы в FAQ.")
        
        with gr.Row():
            with gr.Column():
                q_inp = gr.Textbox(label="Вопрос", placeholder="Как изменить email?")
                a_inp = gr.Textbox(label="Ответ", placeholder="Перейдите в настройки профиля...", lines=3)
                add_btn = gr.Button("Добавить FAQ", variant="primary")
        
        output = gr.Textbox(label="Статус операции")
        faq_display = gr.Textbox(label="Текущие FAQ", value="\n".join([f"В: {faq['question']}\nО: {faq['answer']}\n" for faq in load_faq()]), lines=10)

        add_btn.click(add_faq, inputs=[q_inp, a_inp], outputs=[output, faq_display])

    with gr.Tab("Настройки"):
        gr.Markdown("### Текущие настройки системы")
        gr.Markdown(f"**🦙 API Провайдер:** Ollama (локально)")
        gr.Markdown(f"**Модель:** {MODEL_NAME}")
        gr.Markdown(f"**Ollama Host:** {OLLAMA_HOST}")
        gr.Markdown(f"**Порт:** {PORT}")
        
        gr.Markdown("### Инструкции")
        gr.Markdown("""
        1. **Ollama должен быть запущен:** `ollama serve`
        2. **Модель должна быть создана:** `ollama create support-bot -f Modelfile`
        3. **Добавляйте FAQ** через вкладку "FAQ база"
        4. **Дообучайте модель** новыми данными
        """)

    with gr.Tab("Логи"):
        gr.Markdown("История всех обращений к боту")
        logs_html = gr.HTML(render_logs())
        clear_logs_btn = gr.Button("Очистить логи", variant="stop")
        refresh_logs_btn = gr.Button("Обновить логи")
        
        def update_logs():
            return render_logs()
        
        clear_logs_btn.click(clear_logs, outputs=output)
        refresh_logs_btn.click(update_logs, outputs=logs_html)

def check_ollama_connection():
    """Проверяет подключение к Ollama"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    # Проверяем подключение к Ollama
    if check_ollama_connection():
        print("✅ Ollama подключен и работает")
    else:
        print("⚠️  ВНИМАНИЕ: Ollama недоступен!")
        print("Запустите Ollama: ollama serve")
    
    # Запускаем приложение
    print(f"🚀 Запуск RAG Support Bot на порту {PORT}")
    print(f"🌐 Откройте браузер: http://localhost:{PORT}")
    demo.launch(server_name="0.0.0.0", server_port=PORT)
