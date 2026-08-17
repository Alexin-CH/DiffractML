# src/main.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import train_model, extract_from_batch

from models.linear import LinearModel
from models.conv import FFT_Field_Predictor as ConvNet
from hyper_network import HyperNetwork

from dataset import OpticalSimulationDatasetPD
from utils import LoggerManager
from visualization import display_loss, display_fields, display_difference

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
    title = f"{date}_cnn_model"
    lm = LoggerManager(log_dir=f"{current_dir}/logs", log_name=title)
    lm.logger.info("Logger initialized")

    # Setting device
    cuda_device = 0
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    lm.logger.info(f"Using device: {device}")

    if deterministic:
        fixed_seed_deterministic_run(42)
        lm.logger.info("Fixed seed set for deterministic run")

    lm.logger.info("Initializing model...")

    # Initialize models with corrected channel sizes
    parent_model = LinearModel(in_ch=4).to(device)
    child_model = ConvNet(in_ch=2, hidden=32, out_ch=12).to(device)

    if False:  # Set to True to load pre-trained models
        lm.logger.info("Loading pre-trained models...")
        pretrain_date = "250911_234255"
        child_model = child_model.load_model(f"{current_dir}/model_pt/{pretrain_date}_cnn_model.pt.child.pt", mode='train')
        parent_model = parent_model.load_model(f"{current_dir}/model_pt/{pretrain_date}_cnn_model.pt.parent.pt", mode='train')
        lm.logger.info("Models initialized successfully")

    model = HyperNetwork(parent_model, child_model, mode='first_and_last')

    model = model.to(device)

    lm.logger.info("Loading dataset...")
    dataset = OpticalSimulationDatasetPD(
        data_file="datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pkl",
        fft=True,
        normalize=True,
        device=device,
        logger=lm.logger
    )

    list_epoch_loss, list_val_loss, model = train_model(
        model=model,
        dataset=dataset,
        num_epochs=1e2,
        batch_size=1,
        learning_rate=1e-4,
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

    display_loss(
        list_epoch_loss=list_epoch_loss,
        list_val_loss=list_val_loss,
        title=title,
        dir=current_dir,
        lm=lm
    )

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator(device=device))
        sample = next(iter(loader))
        
        perm_map = dataset.perm_map.to(device)
        perm_map_real = perm_map.real
        perm_map_imag = perm_map.imag

        X_fft = torch.stack([perm_map_real, perm_map_imag], dim=0).unsqueeze(0).float().to(device)
        context, Y_fft = extract_from_batch(sample, device=device)

        model = model.to(device)
        model.eval()
        preds_fft = model(context, X_fft)

        # Invert to get fields from Y
        fields_fft = preds_fft.squeeze(0)
        Ex_fft = torch.complex(fields_fft[0], fields_fft[1])
        Ey_fft = torch.complex(fields_fft[2], fields_fft[3])
        Ez_fft = torch.complex(fields_fft[4], fields_fft[5])
        Hx_fft = torch.complex(fields_fft[6], fields_fft[7])
        Hy_fft = torch.complex(fields_fft[8], fields_fft[9])
        Hz_fft = torch.complex(fields_fft[10], fields_fft[11])

        display_fields(
            Ex=Ex_fft, Ey=Ey_fft, Ez=Ez_fft,
            Hx=Hx_fft, Hy=Hy_fft, Hz=Hz_fft,
            title=f"{title}_fft",
            dir=current_dir,
            lm=lm
        )

        true_fields_fft = Y_fft.squeeze(0)
        Ex_ref_fft = torch.complex(true_fields_fft[0], true_fields_fft[1])
        Ey_ref_fft = torch.complex(true_fields_fft[2], true_fields_fft[3])
        Ez_ref_fft = torch.complex(true_fields_fft[4], true_fields_fft[5])
        Hx_ref_fft = torch.complex(true_fields_fft[6], true_fields_fft[7])
        Hy_ref_fft = torch.complex(true_fields_fft[8], true_fields_fft[9])
        Hz_ref_fft = torch.complex(true_fields_fft[10], true_fields_fft[11])

        display_difference(
            E=Ex_fft, E_ref=Ex_ref_fft,
            H=Hx_fft, H_ref=Hx_ref_fft,
            title=title,
            dir=current_dir,
            lm=lm
        )

if __name__=="__main__":
    main()
    
# end of file
