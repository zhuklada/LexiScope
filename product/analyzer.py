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

print("\nAI analysis:")
print(result)