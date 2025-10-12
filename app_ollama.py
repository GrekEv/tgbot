import os
import json
import time
import pandas as pd
import gradio as gr
import numpy as np
import requests
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# ---------------------------------------
# 1️⃣ Настройки окружения и модели
# ---------------------------------------
# Настройки для Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:8b")
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

# Настройки для OpenAI (если используется)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Устанавливаем API ключ для embeddings (всегда нужен для FAISS)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("🟢 Используется OpenAI для embeddings")
else:
    print("⚠️  ВНИМАНИЕ: OPENAI_API_KEY не установлен для embeddings!")

if USE_OLLAMA:
    print("🦙 Используется локальная Ollama")
else:
    print("🟢 Используется OpenAI API")

DATA_DIR = "data"
FAQ_PATH = os.path.join(DATA_DIR, "faq.csv")
LOG_PATH = os.path.join(DATA_DIR, "logs", "chat_log.json")

# Создаём папку логов, если нет
os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)

# ---------------------------------------
# 2️⃣ Функции для работы с RAG и базой знаний
# ---------------------------------------

def load_faq():
    """Загружает FAQ из CSV файла"""
    if not os.path.exists(FAQ_PATH):
        df = pd.DataFrame(columns=["question", "answer"])
        df.to_csv(FAQ_PATH, index=False)
    return pd.read_csv(FAQ_PATH)

def save_faq(df):
    """Сохраняет FAQ в CSV файл"""
    df.to_csv(FAQ_PATH, index=False)

def build_vectorstore(df):
    """Создаёт FAISS-векторную базу из CSV FAQ"""
    if df.empty:
        # Создаём пустую векторную базу если нет данных
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        store = FAISS.from_texts(["Нет данных"], embeddings)
        store.save_local(os.path.join(DATA_DIR, "faiss_index"))
        return store
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    texts = df["question"].tolist()
    metadatas = [{"answer": a} for a in df["answer"]]
    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    store.save_local(os.path.join(DATA_DIR, "faiss_index"))
    return store

def load_vectorstore():
    """Загружает существующую векторную базу или создаёт новую"""
    if not os.path.exists(os.path.join(DATA_DIR, "faiss_index")):
        df = load_faq()
        return build_vectorstore(df)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.load_local(os.path.join(DATA_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True)

# ---------------------------------------
# 3️⃣ Генерация ответа (RAG + LLM)
# ---------------------------------------

def call_ollama_api(question, context=""):
    """Вызывает локальную Ollama API"""
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Формируем промпт
        if context:
            prompt = f"""Ты профессиональный оператор службы поддержки. На основе контекста ответь на вопрос клиента.

Контекст: {context}

Вопрос клиента: {question}

Ответь профессионально и вежливо на русском языке. Если в контексте нет нужной информации, скажи об этом:"""
        else:
            prompt = f"""Ты профессиональный оператор службы поддержки. Ответь на вопрос клиента: {question}

Ответь вежливо и профессионально на русском языке:"""
        
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
        
    except Exception as e:
        return f"Ошибка при обращении к Ollama: {str(e)}"

def get_rag_response(question, chat_history):
    """Генерирует ответ используя RAG"""
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Получаем релевантные документы
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Если используем Ollama
        if USE_OLLAMA:
            answer = call_ollama_api(question, context)
        else:
            # Используем LangChain с OpenAI
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            answer = result["result"]

        # Логируем
        log_interaction(question, answer)

        return answer
    except Exception as e:
        error_msg = f"Ошибка при генерации ответа: {str(e)}"
        log_interaction(question, error_msg)
        return error_msg

# ---------------------------------------
# 4️⃣ Логирование чата и истории
# ---------------------------------------

def log_interaction(question, answer):
    """Логирует взаимодействие пользователя с ботом"""
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    }
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    logs.append(log)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def load_logs():
    """Загружает логи чата"""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return []

# ---------------------------------------
# 5️⃣ Интерфейс FAQ и админ-функции
# ---------------------------------------

def add_faq(question, answer):
    """Добавляет новый FAQ"""
    if not question.strip() or not answer.strip():
        return "Вопрос и ответ не могут быть пустыми!", load_faq()
    
    df = load_faq()
    new_row = {"question": question, "answer": answer}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_faq(df)
    build_vectorstore(df)
    return "Добавлено успешно!", df

def delete_faq(index):
    """Удаляет FAQ по индексу"""
    df = load_faq()
    if 0 <= index < len(df):
        df = df.drop(index).reset_index(drop=True)
        save_faq(df)
        build_vectorstore(df)
        return "Удалено!", df
    return "Неверный индекс.", df

def refresh_faq():
    """Обновляет список FAQ"""
    return load_faq()

# ---------------------------------------
# 6️⃣ Интерфейс Gradio
# ---------------------------------------

def chat_fn(message, history):
    """Основная функция для вкладки 'Чат'"""
    if not message.strip():
        return history, ""
    
    answer = get_rag_response(message, history)
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
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    return "Логи очищены!"

# Создаём интерфейс
with gr.Blocks(title="RAG Support Bot - Ollama", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Support Bot — панель поддержки с Ollama")
    
    with gr.Tab("Чат"):
        gr.Markdown("Задайте вопрос боту поддержки. Он найдет ответ в базе знаний.")
        chatbot = gr.Chatbot(label="Диалог с ботом", height=400)
        msg = gr.Textbox(label="Введите вопрос", placeholder="Как сбросить пароль?")
        send = gr.Button("Отправить", variant="primary")
        send.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        
        # Обработка Enter
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

    with gr.Tab("FAQ база"):
        gr.Markdown("Управление базой знаний. Добавляйте и удаляйте вопросы из FAQ.")
        
        with gr.Row():
            with gr.Column():
                q_inp = gr.Textbox(label="Вопрос", placeholder="Как изменить email?")
                a_inp = gr.Textbox(label="Ответ", placeholder="Перейдите в настройки профиля...", lines=3)
                add_btn = gr.Button("Добавить FAQ", variant="primary")
            
            with gr.Column():
                delete_index = gr.Number(label="Индекс для удаления", value=0)
                delete_btn = gr.Button("Удалить FAQ", variant="stop")
                refresh_btn = gr.Button("Обновить список")
        
        output = gr.Textbox(label="Статус операции")
        faq_table = gr.Dataframe(load_faq(), label="Текущие вопросы в базе знаний")

        add_btn.click(add_faq, inputs=[q_inp, a_inp], outputs=[output, faq_table])
        delete_btn.click(delete_faq, inputs=[delete_index], outputs=[output, faq_table])
        refresh_btn.click(refresh_faq, outputs=faq_table)

    with gr.Tab("Настройки"):
        gr.Markdown("### Текущие настройки системы")
        
        # Определяем какой API используется
        if USE_OLLAMA:
            gr.Markdown(f"**🦙 API Провайдер:** Ollama (локально)")
            gr.Markdown(f"**Модель:** {MODEL_NAME}")
            gr.Markdown(f"**Ollama Host:** {OLLAMA_HOST}")
            gr.Markdown(f"**Статус:** {'✅ Подключен' if check_ollama_connection() else '❌ Недоступен'}")
        else:
            gr.Markdown(f"**🟢 API Провайдер:** OpenAI")
            gr.Markdown(f"**OpenAI API Key:** {'✅ Установлен' if OPENAI_API_KEY else '❌ Не установлен'}")
        
        gr.Markdown(f"**Путь к данным:** {DATA_DIR}")
        gr.Markdown(f"**Путь к логам:** {LOG_PATH}")
        
        gr.Markdown("### Инструкции по настройке")
        gr.Markdown("""
        **Для Ollama (рекомендуется):**
        1. Установите Ollama: https://ollama.com/download
        2. Запустите: `ollama serve`
        3. Скачайте модель: `ollama pull llama3.2:8b`
        4. Установите `USE_OLLAMA=true`
        
        **Для OpenAI API:**
        1. Установите переменную окружения `OPENAI_API_KEY`
        2. Установите `USE_OLLAMA=false`
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
    if USE_OLLAMA:
        if check_ollama_connection():
            print("✅ Ollama подключен и работает")
        else:
            print("⚠️  ВНИМАНИЕ: Ollama недоступен!")
            print("Запустите Ollama: ollama serve")
    else:
        if not OPENAI_API_KEY:
            print("⚠️  ВНИМАНИЕ: OPENAI_API_KEY не установлен!")
        else:
            print("✅ OpenAI API ключ найден")
    
    # Запускаем приложение
    port = int(os.getenv("PORT", 7860))
    print(f"🚀 Запуск RAG Support Bot на порту {port}")
    print(f"🌐 Откройте браузер: http://localhost:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
