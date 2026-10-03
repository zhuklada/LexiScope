import os
import requests

def analyze_with_ai(text):
    url = "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
    }

    response = requests.post(
        url,
        headers=headers,
        json={"inputs": text}
    )

    return response.json()

review = input("Enter your text: ")

result = analyze_with_ai(review)

sentiment = result[0][0]["label"]
confidence = result[0][0]["score"]

analysis = {
    "sentiment": sentiment,
    "confidence": round(confidence * 100, 2)
}

print("\nAI analysis:")
print(analysis)

words = review.lower().split()
word_count = len(words)

character_count = len(review)

sentence_count = len([s for s in review.split(".") if s.strip()])

if word_count > 0:
    average_word_length = sum(len(word.strip(".,!?;:")) for word in words) / word_count
else:
    average_word_length = 0

result_data = {
    "text": review,
    "sentiment": sentiment,
    "confidence": round(confidence * 100, 2),
    "word_count": word_count,
    "character_count": character_count,
    "sentence_count": sentence_count,
    "average_word_length": round(average_word_length, 2)
}

print("\nLexiScope result:")
print(result_data)

print("\nText statistics:")
print("Word count:", word_count)
print("Character count:", character_count)
print("Sentence count:", sentence_count)
print("Average word length:", round(average_word_length, 2))
import json

with open("analysis_history.json", "a", encoding="utf-8") as file:
    file.write(json.dumps(result_data, ensure_ascii=False) + "\n")

print("\nAnalysis saved!")