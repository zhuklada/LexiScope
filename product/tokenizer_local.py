import json
import re
import unicodedata

class LocalTokenizer:
    def __init__(self, path="tokenizer.json"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = data["model"]["vocab"]
        self.unk_token = data["model"].get("unk_token", "[UNK]")
        self.max_input_chars_per_word = 100

    def tokenize_word(self, word):
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            piece = None

            while start < end:
                text = word[start:end]
                if start > 0:
                    text = "##" + text

                if text in self.vocab:
                    piece = text
                    break

                end -= 1

            if piece is None:
                return [self.unk_token]

            pieces.append(piece)
            start = end

        return pieces

    def tokenize(self, text):
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = "".join(
            " " if unicodedata.category(char) == "Cc"
            else char
            for char in text
        )

        # Split punctuation similarly to BERT's basic tokenizer.
        text = re.sub(r"([!-/:-@\[-`{-~])", r" \1 ", text)

        words = text.strip().split()
        tokens = ["[CLS]"]

        for word in words:
            tokens.extend(self.tokenize_word(word))

        tokens.append("[SEP]")
        return tokens

    def encode(self, text, max_length=128):
        tokens = self.tokenize(text)[:max_length]
        if not tokens:
            tokens = ["[CLS]", "[SEP]"]
        if tokens[-1] != "[SEP]":
            tokens[-1] = "[SEP]"

        input_ids = [self.vocab.get(token, self.vocab["[UNK]"])
                     for token in tokens]
        attention_mask = [1] * len(input_ids)

        padding = max_length - len(input_ids)
        input_ids += [self.vocab["[PAD]"]] * padding
        attention_mask += [0] * padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [0] * max_length,
            "tokens": tokens,
        }

if __name__ == "__main__":
    tokenizer = LocalTokenizer()
    for sample in [
        "This product is amazing!",
        "Продукт ужасен",
    ]:
        result = tokenizer.encode(sample)
        print("\nText:", sample)
        print("Tokens:", result["tokens"])
        print("IDs:", result["input_ids"][:12])