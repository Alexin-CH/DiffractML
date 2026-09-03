import torch
import torch.nn as nn

class RCWA_MLP_correction(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, dtype=torch.float64):
        super(RCWA_MLP_correction, self).__init__()
        self.dtype = dtype
        self.activation = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128, dtype=dtype),
            self.activation,
            nn.Linear(128, 128, dtype=dtype),
            self.activation,
            nn.Linear(128, 64, dtype=dtype),
            self.activation,
            nn.Linear(64, out_ch, dtype=dtype)
        )
    
    def forward(self, s_params):
        x = torch.stack([torch.tensor(v, dtype=torch.complex128) for v in s_params.values()], dim=-1)
        # Extract real and imaginary parts
        x = torch.cat([x.real, x.imag], dim=-1).to(self.dtype)
        return self.net(x).unsqueeze(0)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
        return self

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# end of file
