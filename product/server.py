from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import numpy as np
import onnxruntime as ort
from tokenizer_local import LocalTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tokenizer = LocalTokenizer(
    os.path.join(BASE_DIR, "tokenizer.json")
)

model = ort.InferenceSession(
    os.path.join(BASE_DIR, "model_quantized.onnx"),
    providers=["CPUExecutionProvider"]
)

with open(
    os.path.join(BASE_DIR, "model_config.json"),
    encoding="utf-8"
) as file:
    model_config = json.load(file)

def analyze_sentiment(text):
    encoded = tokenizer.encode(text)

    inputs = {
        name: np.array([encoded[name]], dtype=np.int64)
        for name in (
            "input_ids",
            "attention_mask",
            "token_type_ids"
        )
    }

    logits = model.run(None, inputs)[0][0]
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    if index <= 1:
        sentiment = "NEGATIVE"
    elif index == 2:
        sentiment = "NEUTRAL"
    else:
        sentiment = "POSITIVE"

    return sentiment, round(confidence * 100, 2)

PORT = 8000

def detect_language(text):
    russian = sum(1 for char in text if "а" <= char.lower() <= "я" or char.lower() == "ё")
    english = sum(1 for char in text if "a" <= char.lower() <= "z")

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

    if word_count > 0:
        average_word_length = round(
            sum(len(word.strip(".,!?;:")) for word in words) / word_count,
            2
        )
    else:
        average_word_length = 0

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "average_word_length": average_word_length
    }

def save_history(result):
    history_file = "analysis_history.json"

    with open(history_file, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

class LexiScopeServer(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):

        response = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.end_headers()

        self.wfile.write(response)

    def do_GET(self):

        if self.path == "/":

            try:

                with open(
                    "index.html",
                    "r",
                    encoding="utf-8"
                ) as file:

                    html = file.read().encode("utf-8")

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(html))
                )

                self.end_headers()

                self.wfile.write(html)

            except Exception as error:

                self.send_json(
                    {"error": str(error)},
                    500
                )

            return

        if self.path == "/history":

            history = []

            history_file = "analysis_history.json"

            if os.path.exists(history_file):

                with open(
                    history_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    for line in file:

                        line = line.strip()

                        if not line:
                            continue

                        try:

                            history.append(
                                json.loads(line)
                            )

                        except json.JSONDecodeError:

                            continue

            self.send_json(history)

            return

        self.send_json(
            {"error": "Not found"},
            404
        )

    def do_POST(self):

        if self.path != "/analyze":

            self.send_json(
                {"error": "Not found"},
                404
            )

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            data = json.loads(
                body.decode("utf-8")
            )

            text = data.get(
               "text",
                ""
            ).strip()

            if not text:

                self.send_json(
                    {"error": "Text is empty"},
                    400
                )

                return

            sentiment, confidence = analyze_sentiment(text)

            language = detect_language(text)

            statistics = calculate_statistics(
                text
            )

            result = {

                "text": text,

                "sentiment": sentiment,

                "confidence": confidence,

                "language": language,

                **statistics

            }

            save_history(result)

            self.send_json(result)

        except Exception as error:

            self.send_json(
                {"error": str(error)},
                500
            )

if __name__ == "__main__":

    print(
        f"LexiScope server running on http://localhost:{PORT}"
    )

    server = HTTPServer(
        ("localhost", PORT),
        LexiScopeServer
    )

    server.serve_forever()()