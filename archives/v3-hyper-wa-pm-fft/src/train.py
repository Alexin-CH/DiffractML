# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Casting complex values to real discards the imaginary part")


loss_L1 = nn.SmoothL1Loss()
loss_L2 = nn.MSELoss()
loss_Huber = nn.HuberLoss(delta=0.5)

def extract_from_batch(batch, device):
    """
    sample dict : 'theta' 'wavelength' 'reflectance' 'transmittance' 'fields'
    """

    Ex_fft = batch['Ex_fft_norm'].to(device)
    Ey_fft = batch['Ey_fft_norm'].to(device)
    Ez_fft = batch['Ez_fft_norm'].to(device)
    Hx_fft = batch['Hx_fft_norm'].to(device)
    Hy_fft = batch['Hy_fft_norm'].to(device)
    Hz_fft = batch['Hz_fft_norm'].to(device)

    # Print shapes for debugging
    # print(f"Ex shape: {Ex.shape}, Ey shape: {Ey.shape}, Ez shape: {Ez.shape}")
    # print(f"Hx shape: {Hx.shape}, Hy shape: {Hy.shape}, Hz shape: {Hz.shape}")

    wavelength = batch['wavelength'].to(device)
    angle = batch['theta'].to(device)

    context_wavelength = [wavelength, 1.0 / wavelength]
    context_angle = [torch.sin(angle), torch.cos(angle)]

    fields_fft = torch.stack([Ex_fft.real, Ex_fft.imag,
                        Ey_fft.real, Ey_fft.imag,
                        Ez_fft.real, Ez_fft.imag,
                        Hx_fft.real, Hx_fft.imag,
                        Hy_fft.real, Hy_fft.imag,
                        Hz_fft.real, Hz_fft.imag], dim=-1).permute(0, 3, 1, 2)

    context = torch.stack(context_wavelength + context_angle, dim=-1).float()
    Y_fft = fields_fft.float()
    return context, Y_fft

def train_model(model, dataset, num_epochs, batch_size, learning_rate, device, lm, val_split=0.2):
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator(device=device))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    # Get perm_map from dataset
    perm_map_fft_norm = dataset.perm_map_fft_norm
    perm_map_fft_norm_real = perm_map_fft_norm.real
    perm_map_fft_norm_imag = perm_map_fft_norm.imag
    X = torch.stack([perm_map_fft_norm_real, perm_map_fft_norm_imag]).unsqueeze(0).float().to(device)
    # print(f"Perm_map X shape: {X.shape}")

    # Display the histogram of X
    # import matplotlib.pyplot as plt
    # plt.hist(X.cpu().numpy().flatten(), bins=50)
    # plt.title("Histogram of perm_map_fft_norm values")
    # plt.xlabel("Value")
    # plt.yscale("log")
    # plt.ylabel("Frequency")
    # plt.show()
    # exit(0)

    # optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    # scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs*1, power=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1, min_lr=1e-6)

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
            context, Y_fft= extract_from_batch(batch, device)

            optimizer.zero_grad()
            preds_fft = model(context, X)

            fields_fft = preds_fft.squeeze(0)
            Ex_fft = torch.complex(fields_fft[0], fields_fft[1])
            Ey_fft = torch.complex(fields_fft[2], fields_fft[3])
            Ez_fft = torch.complex(fields_fft[4], fields_fft[5])
            Hx_fft = torch.complex(fields_fft[6], fields_fft[7])
            Hy_fft = torch.complex(fields_fft[8], fields_fft[9])
            Hz_fft = torch.complex(fields_fft[10], fields_fft[11])

            EM_fields = torch.stack([Ex_fft, Ey_fft, Ez_fft, Hx_fft, Hy_fft, Hz_fft], dim=-1)

            fields_true_fft = Y_fft.squeeze(0)
            Ex_true_fft = torch.complex(fields_true_fft[0], fields_true_fft[1])
            Ey_true_fft = torch.complex(fields_true_fft[2], fields_true_fft[3])
            Ez_true_fft = torch.complex(fields_true_fft[4], fields_true_fft[5])
            Hx_true_fft = torch.complex(fields_true_fft[6], fields_true_fft[7])
            Hy_true_fft = torch.complex(fields_true_fft[8], fields_true_fft[9])
            Hz_true_fft = torch.complex(fields_true_fft[10], fields_true_fft[11])

            EM_fields_true = torch.stack([Ex_true_fft, Ey_true_fft, Ez_true_fft, Hx_true_fft, Hy_true_fft, Hz_true_fft], dim=-1)

            loss_amplitude = loss_L2(torch.abs(EM_fields), torch.abs(EM_fields_true))
            loss_phase = loss_Huber(torch.angle(EM_fields), torch.angle(EM_fields_true))

            loss = loss_amplitude + 0.1 * loss_phase
            losses.append(loss)
            loss.backward()

            optimizer.step()

            counter = str(epoch)
            zeros = count - len(counter)
            counter = '0' * zeros + counter
            current_lr = scheduler.get_last_lr()[0]
            tqdm_batchs.set_description(f"[{counter}] Batch Loss: {loss:.6f}, LR: {current_lr:.6f}, Processing")
            lm.logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

        epoch_loss = sum(losses) / len(losses)

        scheduler.step(epoch_loss)
        
        list_epoch_loss.append(epoch_loss.item())
        lm.logger.debug(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {epoch_loss:.6f}")
        tqdm_epochs.set_description(f"Epoch Loss: {epoch_loss:.6f}, Training")

        # Validation
        if (epoch) % 2 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                tqdm_val = tqdm(val_loader, desc="Processing validation batches", leave=False)
                for val_batch in tqdm_val:
                    context_val, Y_fft_val = extract_from_batch(val_batch, device)

                    preds_fft_val = model(context_val, X)

                    val_loss = loss_L2(preds_fft_val, Y_fft_val)

                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            list_val_loss.append(avg_val_loss)
            lm.logger.debug(f"Validation Loss at epoch {epoch + 1}: {avg_val_loss:.6f}")
            model.train()
        else:
            list_val_loss.append(avg_val_loss)

    t1 = time.time() - t0
    th = int(t1 // 3600)
    tm = int((t1 % 3600) // 60)
    ts = int(t1 % 60)
    lm.logger.info(f"Training completed in {th}h {tm}m {ts}s")
    return list_epoch_loss, list_val_loss, model

# end of file
