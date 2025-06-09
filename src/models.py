import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPWithEmbedding(nn.Module):
    def __init__(self, input_dim, num_stations, emb_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_stations, emb_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x_num, station_id):
        emb = self.embedding(station_id)
        x = torch.cat([x_num, emb], dim=1)
        return self.model(x)