exec(open("test_topics_quality.py", encoding="utf-8").read().split("for review, expected in reviews:")[0])

tests = [
    (
        "Доставка опоздала, товар сломан, поддержка не ответила.",
        [
            "The delivery was late.",
            "The product was broken.",
            "Customer support did not reply.",
        ],
    ),
    (
        "Товар отличный, доставка быстрая, поддержка помогла сразу.",
        [
            "The delivery was fast.",
            "The product was excellent.",
            "Customer support helped immediately.",
        ],
    ),
]

for review, hypotheses_list in tests:
    print("\nREVIEW:", review)
    for hypothesis in hypotheses_list:
        print(f"{score(review, hypothesis):.3f} | {hypothesis}")
