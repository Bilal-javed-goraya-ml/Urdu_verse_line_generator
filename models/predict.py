import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.train import Encoder, Decoder
from utils.vocab import Vocab
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_next_masra(input_text):
    vocab = Vocab()
    input_seq = torch.tensor(vocab.encode(input_text)).unsqueeze(0).to(device)

    emb_size = 512
    hidden_size = 512

    encoder = Encoder(len(vocab), emb_size, hidden_size).to(device)
    decoder = Decoder(len(vocab), emb_size, hidden_size).to(device)

    checkpoint = torch.load("trained/model.pt", map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoder_outputs, hidden, cell = encoder(input_seq)

        input_token = torch.tensor([vocab.sos_idx], device=device)
        output_tokens = []

        for _ in range(20):
            output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == vocab.eos_idx:
                break
            output_tokens.append(top1.item())
            input_token = top1

    return vocab.decode(output_tokens)

if __name__ == "__main__":
    input_masra = input("Masra daalein: ")
    print("Agla Masra:")
    print(predict_next_masra(input_masra))
