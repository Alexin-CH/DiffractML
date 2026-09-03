# src/main.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import train_model, extract_from_batch
from models.mlp import RCWA_MLP_correction

from dataset import OpticalSimulationDatasetPD
from utils import LoggerManager
from visualization import display_loss

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


def main(deterministic=False):
    # Setup
    date = datetime.now().strftime("%y%m%d_%H%M%S")
    title = f"{date}_model"
    lm = LoggerManager(log_dir=f"{current_dir}/logs", log_name=title)
    lm.logger.info("Logger initialized")

    # Setting device
    cuda_device = 0
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    lm.logger.info(f"Using device: {device}")

    if deterministic:
        fixed_seed_deterministic_run()
        lm.logger.info("Fixed seed set for deterministic run")

    lm.logger.info("Initializing model...")

    model = RCWA_MLP_correction().to(device)
    lm.logger.info(f"Total number of parameters: {model.num_params()}")

    lm.logger.info("Loading dataset...")
    dataset = OpticalSimulationDatasetPD(
        data_file="datasets/dataset_STT_30nh.csv",
        num_samples=3000,
        device=device,
        logger=lm.logger
    )

    list_epoch_loss, list_val_loss, model = train_model(
        model=model,
        dataset=dataset,
        num_epochs=1e1,
        batch_size=1,
        learning_rate=1e-3,
        val_split=0.2,
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
        lm.logger.error(f"Couldn't save model: {e}")

    display_loss(
        list_epoch_loss=list_epoch_loss,
        list_val_loss=list_val_loss,
        title=title,
        dir=current_dir,
        lm=lm
    )

if __name__=="__main__":
    main()
    
# end of file
