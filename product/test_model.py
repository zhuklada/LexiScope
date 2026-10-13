import json
import numpy as np
import onnxruntime as ort
from tokenizer_local import LocalTokenizer

tokenizer = LocalTokenizer()
session = ort.InferenceSession(
    "model_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

with open("model_config.json", encoding="utf-8") as file:
    config = json.load(file)

tests = [
    "Продукт ужасен",
    "Продукт отличный",
    "Мне не понравилось",
    "Мне очень понравилось",
    "Качество плохое",
    "Качество отличное",
    "The product is terrible",
    "The product is excellent",
]

for text in tests:
    encoded = tokenizer.encode(text)

    inputs = {
        name: np.array([encoded[name]], dtype=np.int64)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }

    logits = session.run(None, inputs)[0][0]
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()

    index = int(np.argmax(probabilities))
    label = config["id2label"][str(index)]

    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {probabilities[index]:.1%}")