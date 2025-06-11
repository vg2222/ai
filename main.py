from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import whisper
import google.generativeai as genai
import time # Добавлен импорт time для замера времени

# ========== НАСТРОЙКИ ==========
UPLOAD_FOLDER   = "uploads"
# ВАЖНО: Замените этот ключ на ваш настоящий ключ Gemini API!
GEMINI_API_KEY  = os.getenv("TOKEN")
HOST            = "0.0.0.0"
PORT            = 5000

# ========== ИНИЦИАЛИЗАЦИЯ ==========
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Конфигурируем Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# Загружаем модель Whisper
# Убедитесь, что установили пакеты: pip install openai-whisper google-generativeai flask flask-cors
# Используйте модель base для более быстрой работы, large для лучшей точности
whisper_model = whisper.load_model("base") 

# Список истории
messages = []


# ========== РОУТЫ ==========

# Главная страница (отдаём index.html)
@app.route("/")
def home():
    # Если у вас есть frontend (index.html), убедитесь, что он находится в той же директории,
    # где запущен Flask-сервер, или укажите путь к нему.
    return send_from_directory(".", "index.html")


# Эндпоинт для загрузки .wav файла
@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"audio_{now}.wav"
    path = os.path.join(UPLOAD_FOLDER, filename)

    print(f"\n[{now}] --- Начат запрос /upload ---")
    print(f"[{now}] Receiving file: {filename}")

    # Сохраняем raw-данные (тело запроса — WAV)
    try:
        with open(path, "wb") as f:
            f.write(request.data)
        print(f"[{now}] File saved successfully: {filename}")
    except Exception as e:
        print(f"[{now}] ERROR: Failed to save file: {e}")
        return jsonify({"status": "error", "error": f"Failed to save file: {str(e)}"})

    # Распознаём речь локально через Whisper
    whisper_start_time = time.time()
    try:
        print(f"[{now}] Starting Whisper transcription for {filename}...")
        result = whisper_model.transcribe(path, language="ru")
        question_text = result["text"].strip()
        whisper_end_time = time.time()
        print(f"[{now}] Whisper transcription complete in {whisper_end_time - whisper_start_time:.2f} seconds.")
    except Exception as e:
        print(f"[{now}] Whisper error: {e}")
        # Удаляем файл, если транскрипция не удалась
        if os.path.exists(path):
            os.remove(path)
            print(f"[{now}] Deleted incomplete file: {filename}")
        return jsonify({"status": "error", "error": f"Whisper transcription failed: {str(e)}"})

    # Запрос в Gemini 2.5 Flash
    gemini_start_time = time.time()
    try:
        print(f"[{now}] Sending question to Gemini: '{question_text}'")
        response = gemini.generate_content(question_text)
        answer_text = response.text.strip()
        gemini_end_time = time.time()
        print(f"[{now}] Gemini response received in {gemini_end_time - gemini_start_time:.2f} seconds.")
    except Exception as e:
        answer_text = "[Ошибка Gemini] " + str(e)
        print(f"[{now}] Gemini error: {e}")

    # Сохраняем в историю
    messages.append({
        "time":     now,
        "question": question_text,
        "answer":   answer_text
    })

    print(f"[{now}] Q: {question_text}")
    print(f"[{now}] A: {answer_text}")
    end_time = time.time()
    print(f"[{now}] --- Запрос /upload завершен за {end_time - start_time:.2f} секунд ---")

    return jsonify({"question": question_text, "answer": answer_text})


# Эндпоинт для ручного текстового запроса
@app.route("/message", methods=["POST"])
def message():
    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    question_text = request.json.get("text", "")
    print(f"\n[{now}] --- Начат запрос /message ---")
    print(f"[{now}] Received text message: '{question_text}'")

    gemini_start_time = time.time()
    try:
        response = gemini.generate_content(question_text)
        answer_text = response.text.strip()
        gemini_end_time = time.time()
        print(f"[{now}] Gemini response received in {gemini_end_time - gemini_start_time:.2f} seconds.")
    except Exception as e:
        answer_text = "[Ошибка Gemini] " + str(e)
        print(f"[{now}] Gemini error: {e}")

    messages.append({
        "time":     now,
        "question": question_text,
        "answer":   answer_text
    })
    end_time = time.time()
    print(f"[{now}] Q: {question_text}")
    print(f"[{now}] A: {answer_text}")
    print(f"[{now}] --- Запрос /message завершен за {end_time - start_time:.2f} секунд ---")
    return jsonify({"answer": answer_text})


# Эндпоинт для истории
@app.route("/history", methods=["GET"])
def history():
    return jsonify(messages)


# Если нужно доставать сохранённые файлы (необязательно)
@app.route("/uploads/<filename>")
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    print(f"Сервер запущен: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False) # debug=False в продакшене
