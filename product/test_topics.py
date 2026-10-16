import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
model = ort.InferenceSession(
    "onnx/model_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

review = "Товар оказался сломанным, но доставка была быстрой, а поддержка ответила вежливо."

topics = [
    "Отзыв о проблемах с доставкой.",
    "Отзыв о качестве товара.",
    "Отзыв о работе поддержки.",
    "Отзыв о цене товара."
]

for topic in topics:
    text = f"Review: {review} Topic: {topic}"
    ids = tokenizer.encode(text).ids
    inputs = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
    }
    logits = model.run(None, inputs)[0][0]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    print(f"{topic} -> entailment: {probs[1]:.3f}")
