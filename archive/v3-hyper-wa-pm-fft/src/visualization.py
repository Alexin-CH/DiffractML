# src/visualization.py

import torch
import os
import matplotlib.pyplot as plt


def display_loss(list_epoch_loss, list_val_loss, title, dir, lm, epoch_min=2):
    lm.logger.info("Plotting loss curve...")
    # Plotting loss curve with linear and log scale
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(list_epoch_loss[epoch_min:], label='Training Loss')
    plt.plot(list_val_loss[epoch_min:], '-r', label='Validation Loss')
    plt.title(f"Loss Curve - {title}")  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(list_epoch_loss[epoch_min:], label='Training Loss')
    plt.plot(list_val_loss[epoch_min:], '-r', label='Validation Loss') 
    plt.xlabel('Epoch')
    plt.xscale('log')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    loss_plot_path = f"{dir}/loss_img/{title}_loss.png"
    os.makedirs(f"{dir}/loss_img", exist_ok=True)
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
    lm.logger.info(f"Loss curve saved as {loss_plot_path}")
    plt.show()

def display_fields(Ex, Ey, Ez, Hx, Hy, Hz, title, dir, lm):
    lm.logger.info("Displaying fields...")
    Enorm = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
    Hnorm = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)

    fig, axes = plt.subplots(figsize=(10,12),nrows=2,ncols=4)
    im0 = axes[0,0].imshow(torch.transpose(Enorm,-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[0,0].set(title='E norm',xlabel='y (nm)',ylabel='z (nm)')
    im1 = axes[0,1].imshow(torch.transpose(torch.abs(Ex),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[0,1].set(title='Ex abs',xlabel='y (nm)',ylabel='z (nm)')
    im2 = axes[0,2].imshow(torch.transpose(torch.abs(Ey),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[0,2].set(title='Ey abs',xlabel='y (nm)',ylabel='z (nm)')
    im3 = axes[0,3].imshow(torch.transpose(torch.abs(Ez),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[0,3].set(title='Ez abs',xlabel='y (nm)',ylabel='z (nm)')
    
    im4 = axes[1,0].imshow(torch.transpose(Hnorm,-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[1,0].set(title='H norm',xlabel='y (nm)',ylabel='z (nm)')
    im5 = axes[1,1].imshow(torch.transpose(torch.abs(Hx),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[1,1].set(title='Hx abs',xlabel='y (nm)',ylabel='z (nm)')
    im6 = axes[1,2].imshow(torch.transpose(torch.abs(Hy),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[1,2].set(title='Hy abs',xlabel='y (nm)',ylabel='z (nm)')
    im7 = axes[1,3].imshow(torch.transpose(torch.abs(Hz),-2,-1).cpu(),cmap='jet',origin='lower',)
    axes[1,3].set(title='Hz abs',xlabel='y (nm)',ylabel='z (nm)')
    
    fig.colorbar(im0,ax=axes[0,0])
    fig.colorbar(im1,ax=axes[0,1])
    fig.colorbar(im2,ax=axes[0,2])
    fig.colorbar(im3,ax=axes[0,3])
    fig.colorbar(im4,ax=axes[1,0])
    fig.colorbar(im5,ax=axes[1,1])
    fig.colorbar(im6,ax=axes[1,2])
    fig.colorbar(im7,ax=axes[1,3])
    plt.tight_layout()
    fields_plot_path = f"{dir}/fields_img/{title}_fields.png"
    os.makedirs(f"{dir}/fields_img", exist_ok=True)
    plt.savefig(fields_plot_path, bbox_inches='tight', dpi=300)
    lm.logger.info(f"Fields plot saved as {fields_plot_path}")
    plt.show()

def display_difference(E, E_ref, H, H_ref, title, dir, lm):
    """
    Display both fields and the difference between two fields.
    
    Parameters:
        F (torch.Tensor): The field to compare.
        F_ref (torch.Tensor): The reference field.
        title (str): Title for the plot.
        lm: Logger manager for logging.
    """
    lm.logger.info("Displaying difference between fields...")
    if E.dtype == torch.complex64 or E.dtype == torch.complex128:
        E_norm = torch.sqrt(torch.abs(E)**2)
    else:
        E_norm = E

    if E_ref.dtype == torch.complex64 or E_ref.dtype == torch.complex128:
        E_ref_norm = torch.sqrt(torch.abs(E_ref)**2)
    else:
        E_ref_norm = E_ref

    E_diff_norm = torch.abs(E_norm - E_ref_norm)

    if H.dtype == torch.complex64 or H.dtype == torch.complex128:
        H_norm = torch.sqrt(torch.abs(H)**2)
    else:
        H_norm = H

    if H_ref.dtype == torch.complex64 or H_ref.dtype == torch.complex128:
        H_ref_norm = torch.sqrt(torch.abs(H_ref)**2)
    else:
        H_ref_norm = H_ref

    H_diff_norm = torch.abs(H_norm - H_ref_norm)

    fig, axes = plt.subplots(figsize=(10, 8), nrows=2, ncols=3)
    im0 = axes[0, 0].imshow(torch.transpose(E_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[0, 0].set(title='E norm', xlabel='y (nm)', ylabel='z (nm)')
    im1 = axes[0, 1].imshow(torch.transpose(E_ref_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[0, 1].set(title='E ref norm', xlabel='y (nm)', ylabel='z (nm)')
    im2 = axes[0, 2].imshow(torch.transpose(E_diff_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[0, 2].set(title='E difference norm', xlabel='y (nm)', ylabel='z (nm)')

    im3 = axes[1, 0].imshow(torch.transpose(H_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[1, 0].set(title='H norm', xlabel='y (nm)', ylabel='z (nm)')
    im4 = axes[1, 1].imshow(torch.transpose(H_ref_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[1, 1].set(title='H ref norm', xlabel='y (nm)', ylabel='z (nm)')
    im5 = axes[1, 2].imshow(torch.transpose(H_diff_norm, -2, -1).cpu(), cmap='jet', origin='lower')
    axes[1, 2].set(title='H difference norm', xlabel='y (nm)', ylabel='z (nm)')

    fig.colorbar(im0, ax=axes[0, 0])
    fig.colorbar(im1, ax=axes[0, 1])
    fig.colorbar(im2, ax=axes[0, 2])
    fig.colorbar(im3, ax=axes[1, 0])
    fig.colorbar(im4, ax=axes[1, 1])
    fig.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    diff_plot_path = f"{dir}/difference_img/{title}_difference.png"
    os.makedirs(f"{dir}/difference_img", exist_ok=True)
    plt.savefig(diff_plot_path, bbox_inches='tight', dpi=300)
    lm.logger.info(f"Difference plot saved as {diff_plot_path}")
    plt.show()

# end of file
