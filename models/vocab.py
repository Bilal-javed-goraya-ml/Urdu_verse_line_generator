import re
from collections import Counter

class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.idx2word = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

        self._build_vocab()

    def _build_vocab(self):
        import pandas as pd
        from glob import glob
        files = glob("data/*.csv")
        # files = glob("data/train.csv")
        counter = Counter()

        for file in files:
            df = pd.read_csv(file)
            for line in pd.concat([df['input'], df['output']]):
                tokens = self.tokenize(line)
                counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def tokenize(self, text):
        text = re.sub(r"[^\u0600-\u06FF\s]", "", text)  # Urdu chars + space
        return text.strip().split()

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.sos_idx] + [self.word2idx.get(t, self.unk_idx) for t in tokens] + [self.eos_idx]

    def decode(self, indices):
        words = [self.idx2word[i] for i in indices if i not in {self.pad_idx, self.sos_idx, self.eos_idx}]
        return " ".join(words)

    def __len__(self):
        return len(self.idx2word)
