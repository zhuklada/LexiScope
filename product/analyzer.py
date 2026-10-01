def analyze_text(text):
    words = text.lower().split()

    return {
        "text": text,
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?"))
    }

review = input("Enter your text: ")

result = analyze_text(review)

print(result)