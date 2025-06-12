import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn

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