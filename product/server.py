from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8000

MODEL_DIR = os.path.join(BASE_DIR, "sentiment_v2")
MODEL_PATH = os.path.join(
    MODEL_DIR, "onnx", "model_quantized.onnx"
)
TOKENIZER_PATH = os.path.join(
    MODEL_DIR, "tokenizer.json"
)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
HISTORY_PATH = os.path.join(BASE_DIR, "analysis_history.json")
HTML_PATH = os.path.join(BASE_DIR, "index.html")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

model = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

with open(CONFIG_PATH, encoding="utf-8") as file:
    model_config = json.load(file)

LABELS = model_config["id2label"]


def analyze_sentiment(text):
    encoded = tokenizer.encode(text)

    inputs = {
        "input_ids": np.array(
            [encoded.ids], dtype=np.int64
        ),
        "attention_mask": np.array(
            [encoded.attention_mask], dtype=np.int64
        ),
    }

    logits = model.run(["logits"], inputs)[0][0]
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    scores = {
        LABELS[str(index)].upper(): float(probability)
        for index, probability in enumerate(probabilities)
    }

    sentiment = max(scores, key=scores.get)
    confidence = scores[sentiment] * 100

    return sentiment, round(confidence, 2), {
        label: round(score * 100, 2)
        for label, score in scores.items()
    }


def detect_language(text):
    russian = sum(
        1 for char in text
        if "а" <= char.lower() <= "я" or char.lower() == "ё"
    )
    english = sum(
        1 for char in text
        if "a" <= char.lower() <= "z"
    )

    if russian > english:
        return "Russian"
    if english > russian:
        return "English"
    return "Unknown"


def calculate_statistics(text):
    words = text.split()
    word_count = len(words)
    character_count = len(text)

    sentences = (
        text.replace("!", ".")
        .replace("?", ".")
        .split(".")
    )
    sentence_count = len(
        [sentence for sentence in sentences if sentence.strip()]
    )

    if word_count:
        average_word_length = round(
            sum(len(word.strip(".,!?;:")) for word in words)
            / word_count,
            2
        )
    else:
        average_word_length = 0

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "average_word_length": average_word_length,
    }


def save_history(result):
    with open(HISTORY_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


class LexiScopeServer(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(
            data, ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        if self.path == "/":
            try:
                with open(HTML_PATH, "r", encoding="utf-8") as file:
                    html = file.read().encode("utf-8")

                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/html; charset=utf-8"
                )
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            except Exception as error:
                self.send_json({"error": str(error)}, 500)
            return

        if self.path == "/history":
            history = []

            if os.path.exists(HISTORY_PATH):
                with open(
                    HISTORY_PATH, "r", encoding="utf-8"
                ) as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            self.send_json(history)
            return

        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/analyze":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", 0)
            )
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            text = data.get("text", "").strip()

            if not text:
                self.send_json({"error": "Text is empty"}, 400)
                return

            sentiment, confidence, scores = analyze_sentiment(text)
            language = detect_language(text)
            statistics = calculate_statistics(text)

            result = {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
                "language": language,
                **statistics,
            }

            save_history(result)
            self.send_json(result)

        except Exception as error:
            self.send_json({"error": str(error)}, 500)


if __name__ == "__main__":
    print(f"LexiScope server running on http://localhost:{PORT}")
    server = HTTPServer(("localhost", PORT), LexiScopeServer)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLexiScope server stopped.")
        server.server_close()
