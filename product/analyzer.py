positive_words = {
    "amazing", "good", "great", "excellent", "love",
    "awesome", "perfect", "nice", "helpful", "happy"
}

negative_words = {
    "bad", "terrible", "awful", "hate", "poor",
    "horrible", "worst", "disappointing", "sad", "useless"
}

def analyze_text(text):
    words = text.lower().split()

    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)

    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?")),
        "positive_words": positive_count,
        "negative_words": negative_count,
        "sentiment": sentiment
    }

review = input("Enter your text: ")

result = analyze_text(review)

print(result)