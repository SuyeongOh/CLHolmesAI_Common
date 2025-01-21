import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        residual = self.skip_conv(x)
        x = self.conv(x)
        x = self.activation(x + residual)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class NIMA(nn.Module):
    def __init__(self, leads):
        super(NIMA, self).__init__()
        self.signal_branch = nn.Sequential(
            ConvBlock(len(leads), 72, 15, 0.2),
            ConvBlock(72, 144, 3, 0.2),
            ConvBlock(144, 288, 5, 0.2),
            ConvBlock(288, 576, 7, 0.2),
        )
        self.fft_branch = nn.Sequential(
            nn.Conv1d(2 * len(leads), 72, kernel_size=3, padding='same'),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),
            nn.Conv1d(72, 144, kernel_size=5, padding='same'),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),
            nn.Conv1d(144, 288, kernel_size=7, padding='same'),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),
            nn.Conv1d(288, 576, kernel_size=9, padding='same'),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),
            nn.Conv1d(576, 1152, kernel_size=11, padding='same'),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(576 + 1152, 576),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(576, 26),
        )

    def forward(self, signal, fft):
        x = self.signal_branch(signal).squeeze(-1)
        y = self.fft_branch(fft).squeeze(-1)
        f = torch.cat([x, y], dim=1)
        f = self.fc(f)
        return torch.sigmoid(f)

if __name__ == '__main__':
    # Model creation example
    leads = 12  # Number of leads
    model = NIMA(leads)

    # Example forward pass
    signal_input = torch.randn(32, leads, 2000)  # Batch of 32, 12 leads, 2000 time steps
    fft_input = torch.randn(32, 2 * leads, 2000)
    output = model(signal_input, fft_input)

    print(output.shape)  # Should be [32, 26]