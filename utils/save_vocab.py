import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json


def save_vocab_json(vocab):
    """Save vocabulary mappings to JSON files"""
    os.makedirs("data", exist_ok=True)
    
    # Save word2idx
    with open("data/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, ensure_ascii=False, indent=2)
    
    # Save idx2word
    with open("data/idx2word.json", "w", encoding="utf-8") as f:
        json.dump(vocab.idx2word, f, ensure_ascii=False, indent=2)
    
    print(" Vocabulary saved to data/word2idx.json and data/idx2word.json")


