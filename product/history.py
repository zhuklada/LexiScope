import json

with open("analysis_history.json", "r", encoding="utf-8") as file:
    history = [json.loads(line) for line in file if line.strip()]

print("\n=== LexiScope History ===\n")

for number, item in enumerate(history, 1):
    print(
        f"{number}. {item['sentiment']} | "
        f"{item['confidence']}% | "
        f"{item['word_count']} words"
    )