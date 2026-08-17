# src/main.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import train_model, extract_from_batch
from models.em_model import EMFieldModel

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

    model = EMFieldModel()

    model = model.to(device)

    lm.logger.info("Loading dataset...")
    dataset = OpticalSimulationDatasetPD(
        data_file="datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pkl",
        normalize=True,
        device=device,
        logger=lm.logger
    )

    list_epoch_loss, list_val_loss, model = train_model(
        model=model,
        dataset=dataset,
        num_epochs=1e3,
        batch_size=32,
        learning_rate=1e-2,
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

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator(device=device))
        sample = next(iter(loader))
        
        perm_map = dataset.perm_map.to(device)
        perm_map_real = perm_map.real
        perm_map_imag = perm_map.imag

        X = torch.stack([perm_map_real, perm_map_imag], dim=0).unsqueeze(0).float().to(device)
        context, Y = extract_from_batch(sample, device=device)

        model = model.to(device)
        model.eval()
        preds = model(context, X)

        # Invert to get fields from Y
        fields = preds.squeeze(0)
        Ex = torch.complex(fields[0], fields[1])
        Ey = torch.complex(fields[2], fields[3])
        Ez = torch.complex(fields[4], fields[5])
        Hx = torch.complex(fields[6], fields[7])
        Hy = torch.complex(fields[8], fields[9])
        Hz = torch.complex(fields[10], fields[11])

        display_fields(
            Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz,
            title=title,
            dir=current_dir,
            lm=lm
        )

        fields_ref = Y.squeeze(0)
        Ex_ref = torch.complex(fields_ref[0], fields_ref[1])
        Ey_ref = torch.complex(fields_ref[2], fields_ref[3])
        Ez_ref = torch.complex(fields_ref[4], fields_ref[5])
        Hx_ref = torch.complex(fields_ref[6], fields_ref[7])
        Hy_ref = torch.complex(fields_ref[8], fields_ref[9])
        Hz_ref = torch.complex(fields_ref[10], fields_ref[11])

        E = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
        E_ref = torch.sqrt(torch.abs(Ex_ref)**2 + torch.abs(Ey_ref)**2 + torch.abs(Ez_ref)**2)

        H = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)
        H_ref = torch.sqrt(torch.abs(Hx_ref)**2 + torch.abs(Hy_ref)**2 + torch.abs(Hz_ref)**2)

        display_difference(
            E=E, E_ref=E_ref,
            H=H, H_ref=H_ref,
            title=title,
            dir=current_dir,
            lm=lm
        )

if __name__=="__main__":
    main()
    
# end of file
