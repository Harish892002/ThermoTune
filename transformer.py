import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
import csv


# =========================
# CONFIG
# =========================
class Config:
    FILENAME = 'rna_sequences_mfe.csv'
    SEQ_LEN = 128   # Matches paragraph

    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 1024
    dropout = 0.1

    batch_size = 128
    lr = 3e-4
    epochs = 25
    weight_decay = 1e-4
    seed = 42

    TEST_PREDICTIONS_CSV = "transformer_test_predictions.csv"
    MUTATION_SCAN_CSV = "transformer_mutation_scan.csv"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on Mac MPS (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on CPU")


torch.manual_seed(Config.seed)
np.random.seed(Config.seed)


# =========================
# DATASET (ONE-HOT 128x4)
# =========================
class RNADataset(Dataset):
    def __init__(self, df, mean=None, std=None):
        self.df = df.reset_index(drop=True)
        self.base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

        if mean is None:
            self.mean = self.df['MFE'].mean()
            self.std = self.df['MFE'].std()
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['Sequence']
        raw_mfe = float(row['MFE'])

        norm_mfe = (raw_mfe - self.mean) / (self.std + 1e-6)

        one_hot = torch.zeros(Config.SEQ_LEN, 4)
        mask = torch.ones(Config.SEQ_LEN, dtype=torch.bool)

        for i, c in enumerate(seq[:Config.SEQ_LEN]):
            if c in self.base_to_idx:
                one_hot[i, self.base_to_idx[c]] = 1.0
                mask[i] = False  # valid position

        return {
            "input": one_hot,
            "mask": mask,
            "y": torch.tensor([norm_mfe], dtype=torch.float32),
            "raw_y": raw_mfe
        }


# =========================
# MODEL
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        scores = self.fc(x).squeeze(-1)
        scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


class RnaTransformerRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(4, Config.d_model)

        self.pos = PositionalEncoding(Config.d_model, Config.SEQ_LEN)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            dim_feedforward=Config.dim_feedforward,
            dropout=Config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer, Config.num_layers, enable_nested_tensor=False
        )

        self.pool = AttentionPooling(Config.d_model)

        self.regressor = nn.Sequential(
            nn.Linear(Config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, mask):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.pool(x, mask)
        return self.regressor(x)


# =========================
# MUTATION SCANNING
# =========================
def run_mutation_scanning(model, dataset, num_sequences=3):
    print("\n================= MUTATION SCANNING =================")
    model.eval()
    device = next(model.parameters()).device

    bases = ['A', 'C', 'G', 'U']
    idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

    rows = []
    indices = np.random.choice(len(dataset), min(num_sequences, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Mutation Scan: Sequences"):
        sample = dataset[idx]
        x = sample["input"]
        mask = sample["mask"]
        valid_len = (~mask).sum().item()

        with torch.no_grad():
            base_pred = model(
                x.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device)
            ).item()

        base_kcal = base_pred * dataset.std + dataset.mean

        for pos in tqdm(range(valid_len), desc=f"Seq {idx} Mutations", leave=False):
            ref = idx_to_base[x[pos].argmax().item()]
            for b in bases:
                if b != ref:
                    m = x.clone()
                    m[pos].zero_()
                    m[pos, dataset.base_to_idx[b]] = 1.0

                    with torch.no_grad():
                        p = model(
                            m.unsqueeze(0).to(device),
                            mask.unsqueeze(0).to(device)
                        ).item()

                    kcal = p * dataset.std + dataset.mean
                    rows.append({
                        "Sequence_ID": idx,
                        "Position": pos,
                        "Ref": ref,
                        "Alt": b,
                        "Mutation": f"{ref}{pos}{b}",
                        "Predicted_MFE": kcal,
                        "Delta": kcal - base_kcal
                    })

    pd.DataFrame(rows).to_csv(Config.MUTATION_SCAN_CSV, index=False)
    print(f"Saved mutation scan CSV → {Config.MUTATION_SCAN_CSV}")


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(Config.FILENAME, quoting=csv.QUOTE_NONE)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=Config.seed)
    val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=Config.seed)

    train_ds = RNADataset(train_df)
    val_ds = RNADataset(val_df, train_ds.mean, train_ds.std)
    test_ds = RNADataset(test_df, train_ds.mean, train_ds.std)

    loaders = {
        "train": DataLoader(train_ds, Config.batch_size, shuffle=True),
        "val": DataLoader(val_ds, Config.batch_size),
        "test": DataLoader(test_ds, Config.batch_size)
    }

    model = RnaTransformerRegressor().to(Config.device)
    opt = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    crit = nn.MSELoss()

    best = float("inf")

    for ep in range(Config.epochs):
        model.train()
        for b in tqdm(loaders["train"], desc=f"Epoch {ep+1}/{Config.epochs} [TRAIN]", leave=False):
            opt.zero_grad()
            loss = crit(
                model(b["input"].to(Config.device), b["mask"].to(Config.device)),
                b["y"].to(Config.device)
            )
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b in tqdm(loaders["val"], desc=f"Epoch {ep+1}/{Config.epochs} [VAL]", leave=False):
                val_loss += crit(
                    model(b["input"].to(Config.device), b["mask"].to(Config.device)),
                    b["y"].to(Config.device)
                ).item()

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_transformer.pth")

    # ================= TEST =================
    model.load_state_dict(torch.load("best_transformer.pth"))
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for b in tqdm(loaders["test"], desc="TEST INFERENCE"):
            p = model(
                b["input"].to(Config.device),
                b["mask"].to(Config.device)
            ).cpu().numpy().flatten()
            preds.extend(p)
            trues.extend(b["raw_y"].numpy())

    preds_kcal = np.array(preds) * train_ds.std + train_ds.mean
    trues_kcal = np.array(trues)

    print("\n=== FINAL METRICS ===")
    print("RMSE:", np.sqrt(mean_squared_error(trues_kcal, preds_kcal)))
    print("MAE :", mean_absolute_error(trues_kcal, preds_kcal))
    print("R2  :", r2_score(trues_kcal, preds_kcal))
    print("r   :", pearsonr(trues_kcal, preds_kcal)[0])

    run_mutation_scanning(model, test_ds)


if __name__ == "__main__":
    main()
