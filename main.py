import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. Data Preprocessing
# =======================
class EnergyDataset(Dataset):
    def __init__(self, df):
        self.X = df[[
            'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos',
            'Temperature_norm', 'Humidity_norm', 'Occupancy_norm', 'SquareFootage_norm'
        ]].values.astype(np.float32)
        self.t = (df['Hour'] / 24.0).values.reshape(-1, 1).astype(np.float32)
        self.temp = df['Temperature_norm'].values.reshape(-1, 1).astype(np.float32)
        self.y = df['EnergyConsumption_norm'].values.reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.t[idx], self.temp[idx], self.y[idx]

# =======================
# 2. TFT-AR Components
# =======================
class TemperatureGatedAFU(nn.Module):
    def __init__(self, num_harmonics=4, temp_embed_dim=10):
        super().__init__()
        self.k_vector = nn.Parameter(torch.arange(1, num_harmonics + 1).float().view(1, -1))
        self.temp_encoder = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, temp_embed_dim)
        )
        self.alpha_net = nn.Linear(temp_embed_dim, num_harmonics)
        self.beta_net = nn.Linear(temp_embed_dim, num_harmonics)
        self.gate_network = nn.Sequential(
            nn.Linear(temp_embed_dim + 1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, t, temp):
        t_norm = t / 168.0
        temp_embed = self.temp_encoder(temp)
        alpha_k = self.alpha_net(temp_embed)
        beta_k = self.beta_net(temp_embed)
        harmonics = 2 * torch.pi * t_norm * self.k_vector
        seasonal_component = (alpha_k * torch.sin(harmonics) + beta_k * torch.cos(harmonics)).sum(dim=1, keepdim=True)
        gate_input = torch.cat([t_norm, temp_embed], dim=-1)
        gate_weight = self.gate_network(gate_input)
        return gate_weight * seasonal_component, seasonal_component, gate_weight

class EnergyEventAttention(nn.Module):
    def __init__(self, input_dim, k=2):
        super().__init__()
        self.energy_scorer = nn.Linear(input_dim, 10)
        self.event_embedder = nn.Linear(10, 1)
        self.k = k

    def forward(self, events):
        attn_scores = self.energy_scorer(events)
        topk_scores, topk_indices = torch.topk(attn_scores, k=self.k, dim=-1)
        sparse_mask = torch.zeros_like(attn_scores).scatter_(-1, topk_indices, 1.0)
        selected_events = sparse_mask * attn_scores
        return self.event_embedder(selected_events)

# =======================
# 3. TFT-AR Model
# =======================
class TFTARModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.afu = TemperatureGatedAFU()
        self.event_net = EnergyEventAttention(input_dim=input_dim)
        self.baseline_net = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, x, t, temp):
        temp_component, season_raw, gate = self.afu(t, temp)
        baseline_component = self.baseline_net(x)
        event_component = self.event_net(x)
        output = baseline_component + temp_component + event_component
        return output, baseline_component, temp_component, event_component, season_raw, gate

# =======================
# 4. Training Function with Multi-Scale Regularization
# =======================
def train_model(model, dataloader, epochs=20, lambda_t=0.1, lambda_g=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, t, temp, y in dataloader:
            x, t, temp, y = x, t, temp, y
            pred, _, _, _, _, _ = model(x, t, temp)
            loss = criterion(pred, y)

            # Multi-scale regularization
            spectral_loss = torch.mean(torch.abs(torch.fft.fft(pred.squeeze()) - torch.fft.fft(y.squeeze()))**2)
            total_reg = loss + lambda_t * spectral_loss

            optimizer.zero_grad()
            total_reg.backward()
            grad_penalty = sum(torch.norm(p.grad.detach(), 2)**2 for p in model.parameters() if p.grad is not None)
            total_reg = total_reg + lambda_g * grad_penalty
            optimizer.step()
            total_loss += total_reg.item()

        train_losses.append(total_loss / len(dataloader))
        print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}")
    return train_losses

# =======================
# 5. Prediction + Save Outputs
# =======================
def predict_and_append(df, model, scaler_energy):
    model.eval()
    dataset = EnergyDataset(df)
    loader = DataLoader(dataset, batch_size=512)

    preds, base, temp, event = [], [], [], []
    with torch.no_grad():
        for x, t, temp_in, _ in loader:
            out, b, tp, ev, _, _ = model(x, t, temp_in)
            preds.append(out.numpy())
            base.append(b.numpy())
            temp.append(tp.numpy())
            event.append(ev.numpy())

    df['PredictedConsumption'] = scaler_energy.inverse_transform(np.vstack(preds))
    df['BaselineComponent'] = scaler_energy.inverse_transform(np.vstack(base))
    df['TemperatureComponent'] = scaler_energy.inverse_transform(np.vstack(temp))
    df['EventComponent'] = scaler_energy.inverse_transform(np.vstack(event))
    return df
from sklearn.preprocessing import MinMaxScaler
import numpy as np
train_df = pd.read_excel(data)
# Sort by time for safety
train_df = train_df.sort_values(by=["Facility", "Timestamp"])

# Normalize numerical columns
scaler_temp = MinMaxScaler()
scaler_hum = MinMaxScaler()
scaler_sqft = MinMaxScaler()
scaler_occ = MinMaxScaler()
scaler_energy = MinMaxScaler()

train_df['Temperature_norm'] = scaler_temp.fit_transform(train_df[['Temperature']])
train_df['Humidity_norm'] = scaler_hum.fit_transform(train_df[['Humidity']])
train_df['SquareFootage_norm'] = scaler_sqft.fit_transform(train_df[['SquareFootage']])
train_df['Occupancy_norm'] = scaler_occ.fit_transform(train_df[['Occupancy']])
train_df['EnergyConsumption_norm'] = scaler_energy.fit_transform(train_df[['EnergyConsumption']])

# Time-based cyclical features
train_df['Hour_sin'] = np.sin(2 * np.pi * train_df['Hour'] / 24)
train_df['Hour_cos'] = np.cos(2 * np.pi * train_df['Hour'] / 24)
train_df['DayOfWeek_sin'] = np.sin(2 * np.pi * train_df['DayOfWeek'] / 7)
train_df['DayOfWeek_cos'] = np.cos(2 * np.pi * train_df['DayOfWeek'] / 7)
train_df['Month_sin'] = np.sin(2 * np.pi * train_df['Month'] / 12)
train_df['Month_cos'] = np.cos(2 * np.pi * train_df['Month'] / 12)

train_dataset = EnergyDataset(train_df)
loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
model = TFTARModel(input_dim=10)
train_losses = train_model(model, loader, epochs=30)
result_df = predict_and_append(train_df.copy(), model, scaler_energy)
