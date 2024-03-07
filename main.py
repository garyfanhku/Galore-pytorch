import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from galore import GaLore


# Simple Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        out = self.transformer(src_embed, tgt_embed)
        out = self.fc(out)
        return out


def main():
    vocab_size = 100
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Generate toy data
    seq_length = 20
    num_samples = 1000
    src_data = torch.randint(0, vocab_size, (num_samples, seq_length))
    tgt_data = torch.randint(0, vocab_size, (num_samples, seq_length))
    dataset = TensorDataset(src_data, tgt_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    galore = GaLore(model, rank=4, update_freq=200)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (src, tgt) in enumerate(dataloader):
            optimizer.zero_grad()

            # Shift the source and target sequences by one position
            src_input = src[:, :-1]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src_input, tgt_input)
            loss = nn.functional.cross_entropy(
                output.view(-1, vocab_size), tgt_output.view(-1)
            )
            loss.backward()
            # Update the model parameters using GaLore
            galore.step(lambda lor_grad: optimizer.step(lor_grad))

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )


if __name__ == "__main__":
    main()
