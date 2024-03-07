import torch

from galore import GaLore

model = torch.linear(2, 1)
optimizer = torch.optim.Adam(model.parameters())
galore = GaLore(model, rank=4, update_freq=200)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        galore.step(lambda lor_grad: optimizer.step(lor_grad))