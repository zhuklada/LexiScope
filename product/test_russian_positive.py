exec(open("test_topics_quality.py", encoding="utf-8").read().split("for review, expected in reviews:")[0])

review = "Товар отличный, доставка быстрая, поддержка помогла сразу."

hypotheses = [
    "Отзыв о положительном опыте доставки.",
    "Доставка была быстрой.",
    "Товар оказался отличным.",
    "Покупатель доволен качеством товара.",
    "Поддержка помогла покупателю.",
    "Покупатель доволен работой службы поддержки.",
]

for hypothesis in hypotheses:
    print(f"{score(review, hypothesis):.3f} | {hypothesis}")
