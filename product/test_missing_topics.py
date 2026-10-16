exec(open("test_topics_quality.py", encoding="utf-8").read().split("for review, expected in reviews:")[0])

reviews_to_test = [
    "Товар отличный, очень доволен покупкой.",
    "Доставка опоздала, заказ пришёл поздно.",
    "Поддержка быстро решила мою проблему.",
]

hypotheses = [
    "Отзыв о проблемах с доставкой.",
    "Отзыв о проблемах с качеством товара.",
    "Отзыв о проблемах с работой поддержки.",
]

for review in reviews_to_test:
    print("\nОТЗЫВ:", review)
    for hypothesis in hypotheses:
        print(f"{score(review, hypothesis):.3f} | {hypothesis}")
