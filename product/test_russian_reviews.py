from server import analyze_sentiment

reviews = [
    ("Товар ужасный, никому не советую", "NEGATIVE"),
    ("Качество отвратительное", "NEGATIVE"),
    ("Доставка опоздала на три дня", "NEGATIVE"),
    ("Мне не понравилось качество товара", "NEGATIVE"),
    ("Товар отличный, всё работает прекрасно", "POSITIVE"),
    ("Очень довольна покупкой", "POSITIVE"),
    ("Доставили быстро, спасибо", "POSITIVE"),
    ("Качество превзошло мои ожидания", "POSITIVE"),
    ("Товар обычный, ничего особенного", "NEUTRAL"),
    ("Получил то, что заказывал", "POSITIVE"),
    ("Упаковка нормальная", "NEUTRAL"),
    ("Пока не могу оценить товар", "NEUTRAL"),
]

correct = 0

for text, expected in reviews:
    predicted, confidence, scores = analyze_sentiment(text)
    is_correct = predicted == expected
    correct += is_correct

    print(f"Текст: {text}")
    print(f"Ожидалось: {expected} | Получено: {predicted} | "
          f"Confidence: {confidence}% | "
          f"{'OK' if is_correct else 'ОШИБКА'}")
    print()

print(f"Результат: {correct}/{len(reviews)}")
print(f"Доля совпадений: {correct / len(reviews) * 100:.1f}%")
