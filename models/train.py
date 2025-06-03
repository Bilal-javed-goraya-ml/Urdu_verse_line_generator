import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.vocab import Vocab
import os
import matplotlib.pyplot as plt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class MasraDataset(Dataset):
    def __init__(self, csv_path, vocab):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = self.vocab.encode(self.data.iloc[idx]['input'])
        output_seq = self.vocab.encode(self.data.iloc[idx]['output'])
        return torch.tensor(input_seq), torch.tensor(output_seq)


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# Attention
class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        hidden = hidden[-1].unsqueeze(2)  # (batch, hidden, 1)
        attn_weights = torch.bmm(encoder_outputs, hidden).squeeze(2)  # (batch, seq_len)
        return torch.softmax(attn_weights, dim=1)


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.attention = LuongAttention(hidden_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden, encoder_outputs)  # (batch, seq_len)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        lstm_input = torch.cat((embedded, attn_applied), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = torch.cat((output.squeeze(1), attn_applied.squeeze(1)), dim=1)
        output = self.fc(output)
        return output, hidden, cell

# Training Function
def train_model():
    print("Starting training...")
    
    # Config
    emb_size = 512
    hidden_size = 512
    batch_size = 8
    epochs = 100
    lr = 0.0001

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
    os.makedirs("trained", exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(), 
        "decoder": decoder.state_dict(),
        "vocab_size": len(vocab),
        "emb_size": emb_size,
        "hidden_size": hidden_size
    }, "trained/model.pt")
    
    # Plot and save training loss
    plot_training_loss(training_losses)
    
    print(" Model saved to trained/model.pt")
    print(" Training completed!")

# Collate Function for Padding
def pad_batch(batch):
    inputs, outputs = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    output_lens = [len(seq) for seq in outputs]

    max_input_len = max(input_lens)
    max_output_len = max(output_lens)

    pad = lambda seqs, max_len: [torch.cat([s, torch.full((max_len - len(s),), 0)]) for s in seqs]

    padded_inputs = torch.stack(pad(inputs, max_input_len))
    padded_outputs = torch.stack(pad(outputs, max_output_len))
    return padded_inputs.long(), padded_outputs.long()

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

def plot_training_loss(losses):
    """Plot and save training loss"""
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    plt.title('Training Loss vs Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" Training loss plot saved to images/training_loss.png")

if __name__ == "__main__":
    train_model()
