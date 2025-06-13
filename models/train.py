import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from models.vocab import Vocab
from models.encoder_decoder import Encoder , Decoder ,LuongAttention
from models.train_data import MasraDataset
from utils.collate import pad_batch
from utils.plot import plot_training_loss
from utils.save_vocab import save_vocab_json
import matplotlib.pyplot as plt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training Function
def train_model():
    print("Starting training...")
    
    # Config
    emb_size = 512
    hidden_size = 256
    batch_size = 16
    epochs = 100
    lr = 0.001

    # Load vocabulary and save to JSON
    vocab = Vocab()
    save_vocab_json(vocab)
    
    # Load data
    train_data = MasraDataset('data/train.csv', vocab)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

    # Initialize models
    encoder = Encoder(vocab_size=len(vocab), emb_size=emb_size, hidden_size=hidden_size).to(device)
    decoder = Decoder(vocab_size=len(vocab), emb_size=emb_size, hidden_size=hidden_size).to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Training loop
    training_losses = []
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, hidden, cell = encoder(src)
            input_token = trg[:, 0]
            loss = 0

            for t in range(1, trg.size(1)):
                output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
                loss += criterion(output, trg[:, t])
                input_token = trg[:, t]  # teacher forcing

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            epoch_loss += loss.item() / trg.size(1)

        avg_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("trained_model", exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(), 
        "decoder": decoder.state_dict(),
        "vocab_size": len(vocab),
        "emb_size": emb_size,
        "hidden_size": hidden_size
    }, "trained_model/model.pt")
    
    # Plot and save training loss
    plot_training_loss(training_losses)
    
    print(" Model saved to trained_model/model.pt")
    print(" Training completed!")

if __name__ == "__main__":
    train_model()
