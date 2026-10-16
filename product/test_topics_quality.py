import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
model = ort.InferenceSession(
    "onnx/model_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

reviews = [
    (
        "Доставка опоздала, товар сломан, поддержка не ответила.",
        "negative"
    ),
    (
        "Товар отличный, доставка быстрая, поддержка помогла сразу.",
        "positive"
    ),
    (
        "The delivery was late, and customer support ignored me.",
        "negative"
    ),
    (
        "The product is excellent and delivery was fast.",
        "positive"
    ),
]

topics = ["delivery", "product quality", "customer support"]

hypotheses = {
    "negative": {
        "delivery": "The review describes a problem with delivery.",
        "product quality": "The review describes a problem with product quality.",
        "customer support": "The review describes a problem with customer support.",
    },
    "positive": {
        "delivery": "The review describes a positive experience with delivery.",
        "product quality": "The review describes a positive experience with product quality.",
        "customer support": "The review describes a positive experience with customer support.",
    },
    "neutral": {
        "delivery": "The review neutrally mentions delivery without praise or complaint.",
        "product quality": "The review neutrally mentions product quality without praise or complaint.",
        "customer support": "The review neutrally mentions customer support without praise or complaint.",
    },
}

def score(review, hypothesis):
    text = f"Review: {review} Hypothesis: {hypothesis}"
    encoded = tokenizer.encode(text)
    ids = encoded.ids
    inputs = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
    }
    logits = model.run(None, inputs)[0][0]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return float(probs[1])

for review, expected in reviews:
    print("\n" + "=" * 70)
    print(f"ОТЗЫВ: {review}")
    print(f"ОЖИДАЕМОЕ ОБЩЕЕ НАСТРОЕНИЕ: {expected}")

    for topic in topics:
        scores = {
            sentiment: score(review, hypotheses[sentiment][topic])
            for sentiment in ("negative", "positive", "neutral")
        }
        best = max(scores, key=scores.get)
        values = " | ".join(
            f"{name}: {value:.3f}" for name, value in scores.items()
        )
        print(f"{topic}: {best} ({values})")
