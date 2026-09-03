# src/main.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import train_model
from models.conv import FieldPredictorV0
from dataset import OpticalSimulationDataset
from utils import LoggerManager

current_dir = os.path.dirname(os.path.abspath(__file__))


def fixed_seed_deterministic_run(seed=42):
    """
    Set fixed seed for reproducibility and deterministic behavior.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    # Setup
    date = datetime.now().strftime("%y%m%d_%H%M%S")
    title = f"{date}_cnn_model"
    lm = LoggerManager(log_dir=f"{current_dir}/logs", log_name=title)
    lm.logger.info("Logger initialized")

    # Setting device
    cuda_device = 0
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    lm.logger.info(f"Using device: {device}")

    # fixed_seed_deterministic_run(42)
    # lm.logger.info("Fixed seed set for deterministic run")

    lm.logger.info("Initializing model...")
    model = FieldPredictorV0(12, 12)
    
    lm.logger.info(f"Setting model to {device}...")
    model = model.to(device)

    lm.logger.info("Loading dataset...")
    dataset = OpticalSimulationDataset("datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pt", logger=lm.logger)

    list_epoch_loss, list_val_loss = train_model(
        model=model,
        dataset=dataset,
        num_epochs=1e3,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
        lm=lm
    )

    try:
        lm.logger.info("Saving model to CPU...")
        model.cpu()
        os.makedirs(f"{current_dir}/model_pt", exist_ok=True)
        model.save_model(f"{current_dir}/model_pt/{title}.pt")
        lm.logger.info(f"Model saved successfully as {title}.pt")
    except Exception as e:
        lm.logger.warning(f"Couldn't save model: {e}")

    lm.logger.info("Plotting loss curve...")
    # Plotting loss curve with linear and log scale
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(list_epoch_loss, label='Training Loss')
    plt.plot(*zip(*list_val_loss), '-r', label='Validation Loss')
    plt.title(f"Loss Curve - {title}")  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(list_epoch_loss, label='Training Loss')
    plt.plot(*zip(*list_val_loss), '-r', label='Validation Loss') 
    plt.xlabel('Epoch')
    plt.xscale('log')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    loss_plot_path = f"{current_dir}/loss_img/{title}_loss.png"
    os.makedirs(f"{current_dir}/loss_img", exist_ok=True)
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
    lm.logger.info(f"Loss curve saved as {loss_plot_path}")
    plt.show()


if __name__=="__main__":
    main()
    
# end of file
