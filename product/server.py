from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import requests

PORT = 8000

HF_URL = "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"

def analyze_with_ai(text):
    token = os.environ["HF_TOKEN"]

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(
        HF_URL,
        headers=headers,
        json={"inputs": text}
    )

    return response.json()

def detect_language(text):
    russian = sum(
        1 for char in text.lower()
        if "а" <= char <= "я"
    )

    english = sum(
        1 for char in text.lower()
        if "a" <= char <= "z"
    )

    if russian > english:
        return "Russian"

    if english > 0:
        return "English"

    return "Unknown"

def calculate_statistics(text):
    words = text.split()

    word_count = len(words)
    character_count = len(text)

    sentence_count = len([
        sentence
        for sentence in text.replace("!", ".")
        .replace("?", ".")
        .split(".")
        if sentence.strip()
    ])

    if word_count > 0:
        average_word_length = sum(
            len(word.strip(".,!?;:"))
            for word in words
        ) / word_count
    else:
        average_word_length = 0

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "average_word_length": round(average_word_length, 2)
    }

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

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.end_headers()

        self.wfile.write(response)

    def do_GET(self):

        if self.path == "/history":
            try:
                with open("analysis_history.json", "r", encoding="utf-8") as file:
                    history = [
                        json.loads(line)
                        for line in file
                        if line.strip()
                    ]

                self.send_json(history)

            except FileNotFoundError:
                self.send_json([])

            return

        if self.path == "/":

            try:
                with open(
                    "index.html",
                    "rb"
                ) as file:

                    content = file.read()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(content))
                )

                self.end_headers()

                self.wfile.write(content)

            except FileNotFoundError:
                self.send_error(404,"index.html not found")

            return

        self.send_error(404)

    def do_POST(self):

        if self.path != "/analyze":

            self.send_error(404)

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
                    {
                        "error": "Text is required."
                    },
                    400
                )

                return

            ai_result = analyze_with_ai(text)

            sentiment = ai_result[0][0]["label"]

            confidence = ai_result[0][0]["score"]

            statistics = calculate_statistics(text)
            language = detect_language(text)

            result = {
                "language": language,
                "text": text,

                "sentiment": sentiment,

                "confidence": round(
                    confidence * 100,
                    2
                ),

                **statistics

            }
            
            with open("analysis_history.json", "a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

            self.send_json(result)

        except Exception as error:

            self.send_json(
                {
                    "error": str(error)
                },
                500
            )

if __name__ == "__main__":

    print()

    print("================================")
    print("        LexiScope Server")
    print("================================")
    print()
    print(
        f"Open http://localhost:{PORT}"
    )
    print()
    print("Press Ctrl+C to stop.")
    print()

    server = HTTPServer(
        ("localhost", PORT),
        LexiScopeServer
    )

    server.serve_forever()