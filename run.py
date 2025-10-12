#!/usr/bin/env python3
"""
Скрипт для запуска RAG Support Bot с дополнительными проверками
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Проверяет установлены ли все зависимости"""
    try:
        import gradio
        import langchain
        import faiss
        import openai
        import pandas
        import numpy
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def check_env_vars():
    """Проверяет переменные окружения"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY не установлен!")
        print("Создайте .env файл или установите переменную окружения")
        return False
    
    print("✅ Переменные окружения настроены")
    return True

def check_data_dir():
    """Проверяет наличие папки данных"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        (data_dir / "logs").mkdir()
        print("✅ Создана папка данных")
    else:
        print("✅ Папка данных существует")
    return True

def main():
    """Основная функция запуска"""
    print("🚀 Запуск RAG Support Bot...")
    print("=" * 50)
    
    # Проверки
    if not check_requirements():
        sys.exit(1)
    
    if not check_env_vars():
        print("Продолжаем без API ключа (для тестирования)...")
    
    if not check_data_dir():
        sys.exit(1)
    
    print("=" * 50)
    print("🎯 Все проверки пройдены, запускаем приложение...")
    
    # Запуск приложения
    try:
        from app import demo
        port = int(os.getenv("PORT", 7860))
        print(f"🌐 Приложение доступно по адресу: http://localhost:{port}")
        demo.launch(server_name="0.0.0.0", server_port=port)
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
