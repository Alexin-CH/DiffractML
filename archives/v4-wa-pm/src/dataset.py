import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def normalize_df_columns(df, columns):
    """
    Normalize specified columns in the DataFrame using global mean and std.
    Adds new columns named '{col}_norm'. Returns (df_copy, stats) where stats contains
    no. of records and dtype/device example.
    """
    df = df.copy()
    N = len(df)
    for c in columns:
        norm_col = f'{c}_norm'
        df.loc[:, norm_col] = [None] * N

    # Collect all tensors into a list for computing global mean/std
    values_flatten = df[columns].values.ravel()
    combined = torch.concatenate([torch.tensor(v).flatten() if not isinstance(v, torch.Tensor) else v.flatten() for v in values_flatten])
    mean = combined.mean()
    std = combined.std()
    if std == 0:
        raise ValueError("Standard deviation is zero, cannot normalize.")

    for i in range(N):
        for c in columns:
            t = torch.tensor(df.at[i, c]) if not isinstance(df.at[i, c], torch.Tensor) else df.at[i, c]
            df.at[i, f'{c}_norm'] = (t - mean) / std

    return df, mean, std

class OpticalSimulationDatasetPD(Dataset):
    def __init__(self, data_file=None, transform=None, normalize=False, logger=None, device='cpu', dtype=torch.complex64):
        """
        Args:
            data_file: path to saved dataset file (.pkl).
            transform: optional callable to apply to each sample.
            fft: compute FFTs for perm_map and field arrays when loading.
            normalize: whether to compute & store normalization stats and normalized arrays.
            logger: optional logger instance.
            device: device for tensors ('cpu' or 'cuda').
            dtype: torch dtype for numeric tensors.
        """
        self.transform = transform
        self.logger = logger
        self.df = None
        self.device = torch.device(device)
        self.dtype = dtype

        if data_file:
            self.load_dataset(data_file, normalize=normalize)

    def __len__(self):
        return 0 if self.df is None else len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Avoid index 0 if possible (for any special handling)
        # There's an issue with index 0 in some datasets, idk why.
        # Probably because the permittivity map is store in the first sample,
        # so this particular sample as one additionnal feature and can't be add in multibatch
        if idx == 0:
            idx = np.random.randint(1, len(self.df) - 1)

        sample = self.df.iloc[idx].copy().to_dict()

        # convert any torch tensors back to tensors (they are stored as torch tensors already)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_dataset(self, output_file):
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if not output_file.endswith('.pkl'):
            output_file += ".pkl"

        # DataFrame may contain torch tensors; pandas will pickle them fine.
        self.df.to_pickle(output_file)

        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=self.dtype)

    def load_dataset(self, input_file, normalize=True):
        if not input_file.endswith('.pkl'):
            raise ValueError("Input file must end with .pkl")

        self.df = pd.read_pickle(input_file)

        # perm_map (assumed same shape across records) -> torch tensor
        self.perm_map = self._to_tensor(self.df.iloc[0]['perm_map']).to(self.device)

        fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

        if normalize:
            self.perm_map_norm = (self.perm_map - self.perm_map.mean()) / self.perm_map.std()
            self.df, self.fields_mean, self.fields_std = normalize_df_columns(self.df, fields)

        if self.logger:
            self.logger.info(f"Loaded {len(self.df)} samples from {input_file}")
        else:
            print(f"Loaded {len(self.df)} samples from {input_file}")
        return self
