import torch
from torch import nn


class SentimentClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_one = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=20, num_layers=2, bidirectional=True, dropout=0.2)
        )
        self.layer_last = nn.Sequential(
            nn.Linear(in_features=40, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=3)
        )

    def forward(self, x: torch.Tensor):
        layer_one_data, second = self.layer_one(x)
        final_layer_data = self.layer_last(layer_one_data)
        return final_layer_data
