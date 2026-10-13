import os
import requests

URL = "https://huggingface.co/Xenova/bert-base-multilingual-uncased-sentiment/resolve/main/onnx/model_quantized.onnx"
OUTPUT = "model_quantized.onnx"
CHUNK_SIZE = 1024 * 1024
EXPECTED_SIZE = 168593695

start = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
print(f"Already downloaded: {start} bytes")

with requests.get(
    URL,
    headers={"Range": f"bytes={start}-{EXPECTED_SIZE - 1}"},
    stream=True,
    timeout=(20, 60),
) as response:
    response.raise_for_status()
    if start and response.status_code != 206:
        raise RuntimeError("Server did not resume download; stopping safely")
    mode = "ab" if start else "wb"
    with open(OUTPUT, mode) as file:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                file.write(chunk)
                file.flush()
                start += len(chunk)
                print(f"\rDownloaded: {start}/{EXPECTED_SIZE} bytes", end="", flush=True)

print()
if os.path.getsize(OUTPUT) != EXPECTED_SIZE:
    print("Download incomplete; progress file kept for resuming.")
else:
    print("Download complete.")
