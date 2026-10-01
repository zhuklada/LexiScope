def analyze_text(text):
    words = text.lower().split()

    return {
        "text": text,
        "word_count": len(words)
    }

review = "The product is amazing and the service is good"

result = analyze_text(review)

print(result)