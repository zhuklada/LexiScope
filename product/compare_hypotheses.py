exec(open("test_topics_quality.py", encoding="utf-8").read().split("for review, expected in reviews:")[0])

tests = [
    (
        "Доставка опоздала, товар сломан, поддержка не ответила.",
        [
            "The delivery was late.",
            "The shipment arrived late.",
            "The product was broken.",
            "The item arrived damaged.",
            "Customer support did not reply.",
        ],
    ),
    (
        "Товар отличный, доставка быстрая, поддержка помогла сразу.",
        [
            "The delivery was fast.",
            "The shipment arrived quickly.",
            "The product was excellent.",
            "The item was of good quality.",
            "Customer support helped immediately.",
        ],
    ),
]

for review, hypotheses_list in tests:
    print("\nREVIEW:", review)
    for hypothesis in hypotheses_list:
        print(f"{score(review, hypothesis):.3f} | {hypothesis}")
