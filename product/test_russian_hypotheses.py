exec(open("test_topics_quality.py", encoding="utf-8").read().split("for review, expected in reviews:")[0])

review = "Доставка опоздала, товар сломан, поддержка не ответила."

hypotheses = [
    "Отзыв о проблемах с доставкой.",
    "Покупатель жалуется на опоздание доставки.",
    "Товар оказался сломанным.",
    "Покупатель недоволен качеством товара.",
    "Поддержка не ответила покупателю.",
    "Покупатель жалуется на работу службы поддержки.",
]

for hypothesis in hypotheses:
    print(f"{score(review, hypothesis):.3f} | {hypothesis}")
