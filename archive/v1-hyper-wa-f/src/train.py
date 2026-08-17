import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import time
from tqdm import tqdm

loss_fn = nn.MSELoss()
loss_entropy = nn.CrossEntropyLoss()

def extract_from_batch(batch, device):
    """
    sample dict : 'phi', 'theta', 'psi', 'wavelength', 'reflectance', 'transmittance', 'fields'
    """
    Ex = batch['fields']['Ex'].to(device)
    Ey = batch['fields']['Ey'].to(device)
    Ez = batch['fields']['Ez'].to(device)
    Hx = batch['fields']['Hx'].to(device)
    Hy = batch['fields']['Hy'].to(device)
    Hz = batch['fields']['Hz'].to(device)

    wavelength = batch['wavelength'].to(device)
    angle = batch['theta'].to(device)

    fields = torch.stack([Ex.real, Ex.imag,
                            Ey.real, Ey.imag,
                            Ez.real, Ez.imag,
                            Hx.real, Hx.imag,
                            Hy.real, Hy.imag,
                            Hz.real, Hz.imag], dim=-1).permute(0, 3, 1, 2)

    # Shape: (batch_size, 12, w, h)
    h = fields.shape[3]
    X = fields[:, :, :, :h//2]
    Y = fields[:, :, :, h//2:]
    context = torch.stack([wavelength, angle], dim=-1).float()

    # Visualization
    # import matplotlib.pyplot as plt
    
    # plt.subplot(1,3,1)
    # plt.imshow(fields[0,0].cpu())
    # plt.subplot(1,3,2)
    # plt.imshow(X[0,0].cpu())
    # plt.subplot(1,3,3)
    # plt.imshow(Y[0,0].cpu())
    # plt.tight_layout()
    # plt.suptitle("Field Visualization")
    # plt.colorbar()
    # plt.show()
    # exit()

    return X, Y, context

def train_model(model, dataset, num_epochs, batch_size, learning_rate, device, lm, val_split=0.2):
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator(device=device))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    list_epoch_loss = []
    list_val_loss = []
    count = len(str(int(num_epochs-1)))
    lm.logger.info("Starting model training...")
    lm.logger.info(f"Number of epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")

    t0 = time.time()
    model.train()
    tqdm_epochs = tqdm(range(int(num_epochs)), desc="Training")
    for epoch in tqdm_epochs:
        lm.logger.debug(f"Epoch {epoch + 1}/{num_epochs} started.")
        losses = []

        show_progress = True if epoch % max(num_epochs // 10, 1) == 0 else False
        tqdm_batchs = tqdm(train_loader, desc="Processing batches", leave=show_progress)
        for batch_idx, batch in enumerate(tqdm_batchs):
            X, Y, context = extract_from_batch(batch, device)

            optimizer.zero_grad()
            preds = model(context, X)

            loss = loss_fn(preds, Y)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            counter = str(epoch)
            zeros = count - len(counter)
            counter = '0' * zeros + counter
            tqdm_batchs.set_description(f"[{counter}] Batch Loss: {loss:.6f}, Processing batches")
            lm.logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.6f}")

        epoch_loss = sum(losses) / len(losses)
        list_epoch_loss.append(epoch_loss.item())
        lm.logger.debug(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {epoch_loss:.6f}")
        tqdm_epochs.set_description(f"Epoch Loss: {epoch_loss:.6f}, Training")

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                tqdm_val = tqdm(val_loader, desc="Processing validation batches", leave=False)
                for val_batch in tqdm_val:
                    X_val, Y_val, context_val = extract_from_batch(val_batch, device)

                    preds_val = model(context_val, X_val)
                    val_loss = loss_fn(preds_val, Y_val)
                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            list_val_loss.append((epoch + 1, avg_val_loss))
            lm.logger.debug(f"Validation Loss at epoch {epoch + 1}: {avg_val_loss:.6f}")
            model.train()

    t1 = time.time() - t0
    th = t1 // 3600
    tm = (t1 % 3600) // 60
    ts = t1 % 60
    lm.logger.info(f"Training completed in {th}h {tm}m {ts}s")
    return list_epoch_loss, list_val_loss
