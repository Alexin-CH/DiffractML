import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class OpticalSimulationDataset(Dataset):
    def __init__(self, data_file=None, transform=None, logger=None):
        """
        Args:
            data_file: path to saved dataset file (.pt).
            transform: optional callable to apply to each sample.
            logger: optional logger instance for logging messages.
        """
        self.transform = transform
        self.samples = []  # will hold a list of dicts
        self.perm_map = None
        self.logger = logger  # logger instance
        
        if data_file:
            self.load_dataset(data_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # torch DataLoader may pass a tensor idx
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_dataset(self, output_file):
        """
        Save the dataset to a .pt file (PyTorch pickled dict).
        """
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if not output_file.endswith('.pt'):
            output_file += ".pt"

        data_to_save = {
            'samples': self.samples,
            'perm_map': self.perm_map
        }
        torch.save(data_to_save, output_file)
        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self

    def get_num_values(self, obj):
        """
        Recursively count the number of values in a nested structure.
        """
        if isinstance(obj, torch.Tensor):
            return obj.numel()
        elif isinstance(obj, (list, tuple)):
            return sum(self.get_num_values(v) for v in obj)
        elif isinstance(obj, dict):
            return sum(self.get_num_values(v) for v in obj.values())
        else:
            return 1

    def load_dataset(self, input_file):
        """
        Load the dataset from a .pt file.
        """
        if not input_file.endswith('.pt'):
            raise ValueError("Output file must end with .pt")

        data = torch.load(input_file)
        # Load samples and perm_map into pytorch tensors if they are not already
        self.samples = data['samples']
        # for sample in self.samples:
        #     fields = sample['fields']
        #     fields_names = list(fields.keys())
        #     for field in fields_names:
        #         fields[f"{field}_mean"] = fields[field].mean()
        #         fields[f"{field}_std"] = fields[field].std()
        #         fields[f"{field}_normalized"] = (fields[field] - fields[f"{field}_mean"]) / fields[f"{field}_std"]                

        self.perm_map = data['perm_map']
        # self.perm_map_mean = self.perm_map.mean()
        # self.perm_map_std = self.perm_map.std()
        # self.perm_map_normalized = (self.perm_map - self.perm_map_mean) / self.perm_map_std

        # self.perm_map_fft = torch.fft.fftshift(torch.fft.fft2(self.perm_map))
        # self.perm_map_fft_mean = self.perm_map_fft.abs().mean()
        # self.perm_map_fft_std = self.perm_map_fft.abs().std()
        # self.perm_map_fft_normalized = (self.perm_map_fft - self.perm_map_fft_mean) / self.perm_map_fft_std

        # Get total number of values in the dataset
        total_values = self.get_num_values(self.samples)

        if self.logger:
            self.logger.info(f"Loaded {len(self.samples)} samples from {input_file}")
            self.logger.info(f"Dataset contains {total_values} total values")
        else:
            print(f"Loaded {len(self.samples)} samples from {input_file}")
            print(f"Dataset contains {total_values} total values")
        return self
