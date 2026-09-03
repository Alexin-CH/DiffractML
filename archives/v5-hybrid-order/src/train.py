# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sintin_torcwa import rcwa

import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Casting complex values to real discards the imaginary part")

class RobustMSELoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(RobustMSELoss, self).__init__()
        self.sigma = sigma

    def forward(self, input, target):
        diff = input - target
        loss = torch.log1p(diff ** 2 / self.sigma).mean()
        return loss

loss_L1 = nn.SmoothL1Loss()
loss_L2 = nn.MSELoss()
loss_robust = RobustMSELoss(sigma=0.01)

loss_fn = loss_L2

Y_keys = [
    'forward.transmission.xx', 'forward.transmission.yx', 'forward.transmission.xy', 
    'forward.transmission.yy', 'forward.transmission.pp', 'forward.transmission.sp', 
    'forward.transmission.ps', 'forward.transmission.ss', 'forward.reflection.xx', 
    'forward.reflection.yx', 'forward.reflection.xy', 'forward.reflection.yy', 
    'forward.reflection.pp', 'forward.reflection.sp', 'forward.reflection.ps', 
    'forward.reflection.ss', 'backward.transmission.xx', 'backward.transmission.yx', 
    'backward.transmission.xy', 'backward.transmission.yy', 'backward.transmission.pp', 
    'backward.transmission.sp', 'backward.transmission.ps', 'backward.transmission.ss', 
    'backward.reflection.xx', 'backward.reflection.yx', 'backward.reflection.xy', 
    'backward.reflection.yy', 'backward.reflection.pp', 'backward.reflection.sp', 
    'backward.reflection.ps', 'backward.reflection.ss'
]

def extract_from_batch(batch, device):
    # Here batch size is 1
    args = rcwa.RCWAArgs(
        wl=batch['wl'],
        ang=batch['ang'],
        nh=2,
        discretization=256,
        sin_amplitude=batch['amp'],
        sin_period=batch['per']
    )
    Y = torch.stack([batch[key] for key in Y_keys], dim=-1).to(device)
    Y = torch.cat([Y.real, Y.imag], dim=-1)
    return args, Y

def train_model(model, dataset, num_epochs, batch_size, learning_rate, device, lm, val_split=0.2):
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator(device=device))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs*1, power=2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=2, min_lr=1e-6)

    list_epoch_loss = []
    list_val_loss = []
    count = len(str(int(num_epochs-1)))
    lm.logger.info("Starting model training...")
    lm.logger.info(f"Number of epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    lm.logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    lm.logger.debug(f"Model architecture: {model}")

    t0 = time.time()
    model.train()
    tqdm_epochs = tqdm(range(int(num_epochs)), desc="Training")
    for epoch in tqdm_epochs:
        lm.logger.debug(f"Epoch {epoch + 1}/{num_epochs} started.")
        losses = []

        show_progress = True if epoch % max(num_epochs // 10, 1) == 0 else False
        tqdm_batchs = tqdm(train_loader, desc="Processing batches", leave=show_progress)
        for batch_idx, batch in enumerate(tqdm_batchs):
            args, Y = extract_from_batch(batch, device)

            rcwa_sim, _ = rcwa.setup(args, device=device)
            s_params = rcwa.get_S_parameters(rcwa_sim)

            preds = model(s_params)

            # print(type(preds), type(Y))
            # print(preds.shape, Y.shape)
            # print(preds.dtype, Y.dtype)
            # exit()

            loss = loss_fn(preds, Y)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            counter = str(epoch)
            zeros = count - len(counter)
            counter = '0' * zeros + counter
            current_lr = scheduler.get_last_lr()[0]
            tqdm_batchs.set_description(f"[{counter}] Batch Loss: {loss:.6f}, LR: {current_lr:.6f}, Processing")
            lm.logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

        epoch_loss = sum(losses) / len(losses)

        scheduler.step()

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
                    args, Y_val = extract_from_batch(val_batch, device)
                    rcwa_results_val, _ = rcwa.setup(args, device=device)
                    s_params_val = rcwa.get_S_parameters(rcwa_results_val)
                    preds_val = model(s_params_val)
                    val_loss = loss_fn(preds_val, Y_val)

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
