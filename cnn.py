import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

def generate_single_nt_mutants(seq, alphabet=("A", "C", "G", "U")):
    mutants = []
    seq = seq.upper()
    for i, ref in enumerate(seq):
        for alt in alphabet:
            if alt != ref:
                mut_seq = seq[:i] + alt + seq[i+1:]
                mutants.append((mut_seq, i, ref, alt))
    return mutants

def mutation_scan_cnn_one_sequence(
    seq_id,
    sequence,
    trained_model,
    stats,
    max_len=128,
    use_structure=False,
    structure=None,
    device="cpu",
    true_mfe_kcal=None,
    top_k=5
):
    trained_model.eval()
    # original prediction
    x = one_hot_sequence(sequence, max_len)
    if use_structure and structure is not None:
        struct_x = one_hot_dotbracket(structure, max_len)
        x = np.concatenate([x, struct_x], axis=-1)
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_norm = trained_model(x_t).cpu().item()
    if stats["std_mfe"] is not None:
        pred_kcal = pred_norm * stats["std_mfe"] + stats["mean_mfe"]
    else:
        pred_kcal = pred_norm

    # mutants
    mutants = generate_single_nt_mutants(sequence)
    mut_records = []
    for mut_seq, pos, ref, alt in mutants:
        xm = one_hot_sequence(mut_seq, max_len)
        if use_structure and structure is not None:
            struct_xm = one_hot_dotbracket(structure, max_len)
            xm = np.concatenate([xm, struct_xm], axis=-1)
        xm_t = torch.from_numpy(xm).permute(1, 0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            mut_pred_norm = trained_model(xm_t).cpu().item()
        if stats["std_mfe"] is not None:
            mut_pred_kcal = mut_pred_norm * stats["std_mfe"] + stats["mean_mfe"]
        else:
            mut_pred_kcal = mut_pred_norm
        delta = mut_pred_kcal - pred_kcal  # negative = stabilizing
        mut_records.append({
            "mut_seq": mut_seq,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "pred_mfe": mut_pred_kcal,
            "delta": delta,
        })

    # sort by most stabilizing (lowest ΔG, i.e., most negative change)
    mut_records.sort(key=lambda r: r["delta"])

    # pretty print
    print(f"\nSequence ID {seq_id} (Len {len(sequence)})")
    print(f"Original Seq: {sequence[:10]}...")
    if true_mfe_kcal is not None:
        print(f"  True MFE:      {true_mfe_kcal:8.2f} kcal/mol")
    print(f"  Predicted MFE: {pred_kcal:8.2f} kcal/mol")
    print("Top 5 Stabilizing Mutations:")
    for rank, rec in enumerate(mut_records[:top_k], start=1):
        pos = rec["pos"]
        print(f"  {rank}. {rec['ref']}{pos}{rec['alt']}: {rec['pred_mfe']:8.2f} kcal/mol (Change: {rec['delta']:+.2f})")

# -- One-hot encoding utilities --
NUC_ORDER = ["A", "C", "G", "U"]
NUC_TO_IDX = {n: i for i, n in enumerate(NUC_ORDER)}
DOTBRACKET_SYMBOLS = ["(", ")", ".", "[", "]", "{", "}", "<", ">"]
DB_TO_IDX = {s: i for i, s in enumerate(DOTBRACKET_SYMBOLS)}

def one_hot_sequence(seq, max_len=128):
    arr = np.zeros((max_len, len(NUC_ORDER)), dtype=np.float32)
    L = min(len(seq), max_len)
    for i in range(L):
        nuc = seq[i].upper()
        if nuc in NUC_TO_IDX:
            arr[i, NUC_TO_IDX[nuc]] = 1.0
    return arr

def one_hot_dotbracket(struct, max_len=128):
    arr = np.zeros((max_len, len(DOTBRACKET_SYMBOLS)), dtype=np.float32)
    L = min(len(struct), max_len)
    for i in range(L):
        ch = struct[i]
        if ch in DB_TO_IDX:
            arr[i, DB_TO_IDX[ch]] = 1.0
    return arr

def generate_single_nt_mutants(seq, alphabet=("A", "C", "G", "U")):
    mutants = []
    seq = seq.upper()
    for i, ref in enumerate(seq):
        for alt in alphabet:
            if alt != ref:
                mut_seq = seq[:i] + alt + seq[i+1:]
                mutants.append((mut_seq, i, ref, alt))
    return mutants

def mutation_scan_cnn_one_sequence(
    seq_id,
    sequence,
    trained_model,
    stats,
    max_len=128,
    use_structure=False,
    structure=None,
    device="cpu",
    true_mfe_kcal=None,
    top_k=5
):
    trained_model.eval()
    # original prediction
    x = one_hot_sequence(sequence, max_len)
    if use_structure and structure is not None:
        struct_x = one_hot_dotbracket(structure, max_len)
        x = np.concatenate([x, struct_x], axis=-1)
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_norm = trained_model(x_t).cpu().item()
    if stats["std_mfe"] is not None:
        pred_kcal = pred_norm * stats["std_mfe"] + stats["mean_mfe"]
    else:
        pred_kcal = pred_norm

    # mutants
    mutants = generate_single_nt_mutants(sequence)
    mut_records = []
    for mut_seq, pos, ref, alt in mutants:
        xm = one_hot_sequence(mut_seq, max_len)
        if use_structure and structure is not None:
            struct_xm = one_hot_dotbracket(structure, max_len)
            xm = np.concatenate([xm, struct_xm], axis=-1)
        xm_t = torch.from_numpy(xm).permute(1, 0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            mut_pred_norm = trained_model(xm_t).cpu().item()
        if stats["std_mfe"] is not None:
            mut_pred_kcal = mut_pred_norm * stats["std_mfe"] + stats["mean_mfe"]
        else:
            mut_pred_kcal = mut_pred_norm
        delta = mut_pred_kcal - pred_kcal  # negative = stabilizing
        mut_records.append({
            "mut_seq": mut_seq,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "pred_mfe": mut_pred_kcal,
            "delta": delta,
        })

    # sort by most stabilizing (lowest ΔG, i.e., most negative change)
    mut_records.sort(key=lambda r: r["delta"])

    # pretty print
    print(f"\nSequence ID {seq_id} (Len {len(sequence)})")
    print(f"Original Seq: {sequence[:10]}...")
    if true_mfe_kcal is not None:
        print(f"  True MFE:      {true_mfe_kcal:8.2f} kcal/mol")
    print(f"  Predicted MFE: {pred_kcal:8.2f} kcal/mol")
    print("Top 5 Stabilizing Mutations:")
    for rank, rec in enumerate(mut_records[:top_k], start=1):
        pos = rec["pos"]
        print(f"  {rank}. {rec['ref']}{pos}{rec['alt']}: {rec['pred_mfe']:8.2f} kcal/mol (Change: {rec['delta']:+.2f})")

# -- Stratified length bins and split --
def make_length_bins(lengths, num_bins=5):
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(lengths, quantiles)
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    bins = np.digitize(lengths, bin_edges[1:-1])
    return bins

def train_val_test_split_stratified(df, length_col="seq_len", train_frac=0.8, val_frac=0.1, test_frac=0.1, num_bins=5, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    lengths = df[length_col].values.astype(np.float32)
    y_bins = make_length_bins(lengths, num_bins=num_bins)
    df_train, df_temp, y_train, y_temp = train_test_split(df, y_bins, test_size=(1.0 - train_frac), stratify=y_bins, random_state=random_state, shuffle=True)
    val_ratio_temp = val_frac / (val_frac + test_frac)
    lengths_temp = df_temp[length_col].values.astype(np.float32)
    y_temp_bins = make_length_bins(lengths_temp, num_bins=num_bins)
    df_val, df_test, _, _ = train_test_split(df_temp, y_temp_bins, test_size=(1.0 - val_ratio_temp), stratify=y_temp_bins, random_state=random_state, shuffle=True)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# -- Dataset/dataloader
class RNADataset(Dataset):
    def __init__(self, df, max_len=128, use_structure=True, mean_mfe=None, std_mfe=None):
        self.df = df
        self.max_len = max_len
        self.use_structure = use_structure
        self.mean_mfe = mean_mfe
        self.std_mfe = std_mfe
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq, struct, mfe = row["Sequence"], row["Structure"], float(row["MFE"])
        seq_oh = one_hot_sequence(seq, self.max_len)
        if self.use_structure:
            struct_oh = one_hot_dotbracket(struct, self.max_len)
            x = np.concatenate([seq_oh, struct_oh], axis=-1)
        else:
            x = seq_oh
        if (self.mean_mfe is not None) and (self.std_mfe is not None) and (self.std_mfe > 0):
            y = (mfe - self.mean_mfe) / self.std_mfe
        else:
            y = mfe
        x_tensor = torch.from_numpy(x).permute(1, 0).float()
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return {"inputs": x_tensor, "target": y_tensor, "length": min(len(seq), self.max_len), "raw_mfe": torch.tensor(mfe, dtype=torch.float32)}

def create_dataloaders_from_csv(
    csv_path, batch_size=256, max_len=128, use_structure=True,
    num_bins=5, random_state=42, num_workers=4, pin_memory=True, normalize_mfe=True
):
    df = pd.read_csv(csv_path)
    df["seq_len"] = df["Sequence"].apply(len)
    df_train, df_val, df_test = train_val_test_split_stratified(df, length_col="seq_len", num_bins=num_bins, random_state=random_state)
    if normalize_mfe:
        mean_mfe, std_mfe = df_train["MFE"].mean(), df_train["MFE"].std(ddof=0)
    else:
        mean_mfe, std_mfe = None, None
    train_set = RNADataset(df_train, max_len=max_len, use_structure=use_structure, mean_mfe=mean_mfe, std_mfe=std_mfe)
    val_set = RNADataset(df_val, max_len=max_len, use_structure=use_structure, mean_mfe=mean_mfe, std_mfe=std_mfe)
    test_set = RNADataset(df_test, max_len=max_len, use_structure=use_structure, mean_mfe=mean_mfe, std_mfe=std_mfe)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    stats = {"mean_mfe": mean_mfe, "std_mfe": std_mfe, "n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set)}
    return {"train": train_loader, "val": val_loader, "test": test_loader}, stats

# -- Model (as described in your specs)
class RNACNNRegressor(nn.Module):
    def __init__(self, input_channels=4+9, seq_len=128, conv_drop=0.1, fc_drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=11, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(conv_drop)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(conv_drop)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding="same")
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(fc_drop)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x

# -- Training/validation metric calculation functions
def train_epoch(model, loader, optimizer, device):
    model.train(); running_loss = 0.
    for batch in tqdm(loader, desc="Train"):
        x = batch["inputs"].to(device)
        y = batch["target"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval(); running_loss = 0.; all_preds = []; all_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            running_loss += loss.item() * x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds); targets = np.concatenate(all_targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r = pearsonr(targets, preds)[0]
    r2 = r2_score(targets, preds)
    return running_loss / len(loader.dataset), mse, mae, r, r2, preds, targets

def fit(model, loaders, epochs=40, lr=1e-3, weight_decay=1e-5, device="cuda", early_stop=7):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    best_val, best_ep = float('inf'), 0; best_state = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        val_loss, val_mse, val_mae, val_r, val_r2, _, _ = eval_epoch(model, loaders["val"], device)
        scheduler.step(val_loss)
        print(f"Ep {epoch+1:02d}: train {train_loss:.4f} | val {val_loss:.4f} | val_MSE {val_mse:.4f} | val_MAE {val_mae:.4f} | val_r {val_r:.3f} | val_R2 {val_r2:.3f}")
        if val_loss < best_val:
            best_val, best_ep = val_loss, epoch
            best_state = model.state_dict()
        if epoch - best_ep > early_stop:
            print("Early stopping."); break
    model.load_state_dict(best_state)
    return model

# -- Main Experiment
if __name__ == "__main__":
    csv_path = "rna_sequences_mfe.csv"
    loaders, stats = create_dataloaders_from_csv(
        csv_path, batch_size=256, max_len=128, use_structure=True, num_bins=5, random_state=123, normalize_mfe=True
    )
    print(stats)
    input_channels = loaders["train"].dataset[0]['inputs'].shape[0]
    model = RNACNNRegressor(input_channels=input_channels, seq_len=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = fit(model, loaders, epochs=40, lr=2e-3, weight_decay=1e-6, device=device, early_stop=7)
    print("Best model loaded.")

    model_path = "cnn_rna_model_best.pth"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "stats": stats,               # mean/std for de-normalization
            "input_channels": input_channels,
        },
        model_path,
    )
    print(f"Saved model checkpoint to {model_path}")

    # Test set evaluation (in normalized units)
    test_loss, test_mse, test_mae, test_r, test_r2, preds, targets = eval_epoch(trained_model, loaders["test"], device=device)
    print(f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f} | r: {test_r:.3f} | R2: {test_r2:.3f}")

    output_file = "cnn_test_predictions.csv"
    test_df = loaders["test"].dataset.df.reset_index(drop=True)
    if stats['std_mfe'] is not None:
        preds_kcal = (preds * stats['std_mfe']) + stats['mean_mfe']
        targets_kcal = (targets * stats['std_mfe']) + stats['mean_mfe']
    else:
        preds_kcal = preds
        targets_kcal = targets
    out = pd.DataFrame({
        "Sequence": test_df["Sequence"],
        "Structure": test_df["Structure"],
        "True_MFE": targets_kcal,
        "Pred_MFE": preds_kcal
    })
    out.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}.")

    rmse_kcal = np.sqrt(test_mse)

    print("\n=== FINAL TEST METRICS ===")
    print(f"RMSE (kcal/mol): {rmse_kcal:8.4f}")
    print(f"MAE (kcal/mol):  {test_mae:8.4f}")
    print(f"R2 Score:        {test_r2:8.4f}")
    print(f"Pearson r:       {test_r:8.4f}")

    # Convert to physical units (kcal/mol) if normalized
    stats_str = ""
    if stats['std_mfe'] is not None:
        preds_kcal = (preds * stats['std_mfe']) + stats['mean_mfe']
        targets_kcal = (targets * stats['std_mfe']) + stats['mean_mfe']
        mse_kcal = mean_squared_error(targets_kcal, preds_kcal)
        mae_kcal = mean_absolute_error(targets_kcal, preds_kcal)
        r_kcal = pearsonr(targets_kcal, preds_kcal)[0]
        r2_kcal = r2_score(targets_kcal, preds_kcal)
        stats_str = f" [kcal/mol] MAE: {mae_kcal:.2f}  MSE: {mse_kcal:.2f}  r: {r_kcal:.3f} R2: {r2_kcal:.3f}"
        print("Test set (kcal/mol units):", stats_str)

    print("\n========================================")
    print("   PHASE 2: MUTATION SCANNING")
    print("========================================")

    test_df = loaders["test"].dataset.df.reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pick a few example indices from test set (adapt indices as you like)
    example_indices = [0, 1, 2]  # or random selection, or specific IDs

    for idx in example_indices:
        seq = test_df.loc[idx, "Sequence"]
        struct = test_df.loc[idx, "Structure"] if "Structure" in test_df.columns else None
        # true MFE in kcal/mol is the original MFE column
        true_mfe_kcal = float(test_df.loc[idx, "MFE"])
        mutation_scan_cnn_one_sequence(
            seq_id=idx,
            sequence=seq,
            trained_model=trained_model,
            stats=stats,
            max_len=128,
            use_structure=True,  # or False if sequence-only
            structure=struct,
            device=device,
            true_mfe_kcal=true_mfe_kcal,
            top_k=5,
        )