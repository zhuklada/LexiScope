import hashlib
import os
import time
import urllib.request

URL = "https://huggingface.co/Horizon-Labs/multilingual-sentiment-small/resolve/fb4a16bc41f5807345f17f6d5a81fa5f8760cbb6/onnx/model_quantized.onnx"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "sentiment_v2", "onnx", "model_quantized.onnx.download")
TOTAL = 268410774
EXPECTED_SHA256 = "a2b63c00af1dbb19ddbfdf7376fefb9b4abab6f54819de226f9649d367a9b2d9"
FINAL_OUTPUT = os.path.join(BASE_DIR, "sentiment_v2", "onnx", "model_quantized.onnx")
CHUNK = 1024 * 1024

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
current = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0

if current > TOTAL:
    raise SystemExit(f"Файл больше ожидаемого размера: {current}")

print(f"Уже загружено: {current:,} из {TOTAL:,} байт")

with open(OUTPUT, "ab") as f:
    while current < TOTAL:
        end = min(current + CHUNK - 1, TOTAL - 1)
        request = urllib.request.Request(
            URL,
            headers={"Range": f"bytes={current}-{end}"}
        )

        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f"bytes {current}-{end}/{TOTAL}"
                    actual_range = response.headers.get("Content-Range")

                    if response.status != 206 or actual_range != expected_range:
                        raise RuntimeError(
                            f"Неожиданный ответ: HTTP {response.status}, "
                            f"Content-Range={actual_range}"
                        )

                    data = response.read()
                    if len(data) != end - current + 1:
                        raise RuntimeError("Получен неполный блок")

                f.write(data)
                f.flush()
                current += len(data)
                print(f"\rЗагружено: {current:,}/{TOTAL:,} байт "
                      f"({current / TOTAL:.1%})", end="", flush=True)
                break
            except Exception as error:
                print(f"\nОшибка блока, попытка {attempt}/5: {error}")
                if attempt == 5:
                    raise
                time.sleep(attempt * 2)

print("\nПроверяем SHA-256...")
sha256 = hashlib.sha256()
with open(OUTPUT, "rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
        sha256.update(block)

actual_sha256 = sha256.hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise SystemExit(f"SHA-256 не совпадает: {actual_sha256}")

os.replace(OUTPUT, FINAL_OUTPUT)
print("Загрузка завершена, SHA-256 подтверждён.")
print("Модель сохранена:", FINAL_OUTPUT)
