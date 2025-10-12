#!/usr/bin/env python3
"""
Простой CORS прокси для Ollama API
Запускает локальный сервер для обхода CORS ограничений браузера
"""

import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import sys

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/generate':
            try:
                # Читаем данные запроса
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Парсим JSON
                request_data = json.loads(post_data.decode('utf-8'))
                
                # Формируем запрос к Ollama
                ollama_url = 'http://localhost:11434/api/generate'
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(request_data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                
                # Отправляем запрос к Ollama
                with urllib.request.urlopen(req) as response:
                    response_data = response.read()
                
                # Отправляем ответ клиенту
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
            except Exception as e:
                print(f"Ошибка: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": str(e)})
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/tags':
            try:
                # Проксируем запрос к Ollama
                ollama_url = 'http://localhost:11434/api/tags'
                with urllib.request.urlopen(ollama_url) as response:
                    response_data = response.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
            except Exception as e:
                print(f"Ошибка: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": str(e)})
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    PORT = 8080
    
    print(f"🚀 Запуск CORS прокси на порту {PORT}")
    print(f"🌐 Проксирует запросы к Ollama (localhost:11434)")
    print(f"📡 Доступен по адресу: http://localhost:{PORT}")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
            print(f"✅ Сервер запущен на http://localhost:{PORT}")
            print("🔄 Проксирует запросы к Ollama...")
            print("⏹️  Для остановки нажмите Ctrl+C")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")

if __name__ == "__main__":
    main()
