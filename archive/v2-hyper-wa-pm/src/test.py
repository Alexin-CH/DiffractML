import torch
from matplotlib import pyplot as plt

from models.linear import MLPnet
from dataset_utils_pt import OpticalSimulationDataset


def fields_from_net(preds):
    preds_reshaped = preds.view(preds.shape[0], -1, 12, 256).permute(0, 2, 1, 3).squeeze(0)
    print(f"Preds shape: {preds.shape}, Reshaped shape: {preds_reshaped.shape}")
    # Rebuild fields
    Ex = preds_reshaped[:, 0 :] + 1j * preds_reshaped[:, 1, :]
    Ey = preds_reshaped[:, 2, :] + 1j * preds_reshaped[:, 3, :]
    Ez = preds_reshaped[:, 4, :] + 1j * preds_reshaped[:, 5, :]
    Hx = preds_reshaped[:, 6, :] + 1j * preds_reshaped[:, 7, :]
    Hy = preds_reshaped[:, 8, :] + 1j * preds_reshaped[:, 9, :]
    Hz = preds_reshaped[:, 10, :] + 1j * preds_reshaped[:, 11, :]
    return Ex, Ey, Ez, Hx, Hy, Hz


def main():
    pretrain_date = "250716_155322"
    model = MLPnet(in_ch=256, out_ch=12*256)
    model = model.load_model(f"model_pt/{pretrain_date}_cnn_model.pt", mode='eval')

    dataset = OpticalSimulationDataset("datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_256.pt")

    X = dataset[0]['perm_map'].float()

    Y = model(X)

    with torch.no_grad():
        Ex, Ey, Ez, Hx, Hy, Hz = fields_from_net(Y)

    # Plotting initial Ex,  reconstructed Ex and difference
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    Ex_init = dataset[0]['fields']['Ex'].abs().cpu().numpy()
    plt.imshow(Ex_init, cmap='gray')
    plt.title("Input Permittivity Map")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    Ex_reconstructed = Ex.abs().cpu().numpy()
    plt.imshow(Ex_reconstructed, cmap='gray')
    plt.title("Reconstructed Ex")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    Ex_difference = Ex_init - Ex_reconstructed
    plt.imshow(Ex_difference, cmap='gray')
    plt.title("Difference Ex")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()