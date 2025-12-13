import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & DEVICE SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

BATCH_SIZE = 1 
EPOCHS = 300       
LR = 0.001
EMBED_DIM = 64 

# --- SPARSE ATTENTION CONFIG ---
# Radius in "Normalized Space" (since we use StandardScaler)
# 0.3 means "Only look at hits that are geometrically close to me"
# This physically restricts the attention to local neighborhoods.
SPARSE_RADIUS = 0.3 

# ==========================================
# 2. DATASET LOADER (Standard)
# ==========================================
class ParticleDataset(Dataset):
    def __init__(self, file_path, scaler=None, is_train=True):
        print(f"Loading {file_path}...")
        self.df = pd.read_excel(file_path)
        self.events = self.df['event_id'].unique()
        self.is_train = is_train
        self.features = ['x', 'y', 'z']
        
        if is_train:
            self.scaler = StandardScaler()
            self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
        else:
            self.scaler = scaler
            self.df[self.features] = self.scaler.transform(self.df[self.features])
            
    def __len__(self): return len(self.events)

    def __getitem__(self, idx):
        ev_id = self.events[idx]
        event_df = self.df[self.df['event_id'] == ev_id]
        
        x = torch.tensor(event_df[self.features].values, dtype=torch.float32)
        y = torch.tensor(event_df['track_id'].values, dtype=torch.long)
        w = torch.tensor(calculate_weights(event_df), dtype=torch.float32)
        return x, y, w, ev_id

def calculate_weights(df_event):
    layers = sorted(df_event['layer'].unique())
    inner_layers = layers[:2] if len(layers) > 2 else []
    outer_layers = layers[-2:] if len(layers) > 2 else []
    weights = []
    for _, row in df_event.iterrows():
        w = 1.0
        if row['layer'] in inner_layers or row['layer'] in outer_layers: w *= 2.0
        if row['track_id'] == 0: w = 0.0
        weights.append(w)
    weights = np.array(weights)
    if weights.sum() > 0: weights /= weights.sum()
    return weights

# ==========================================
# 3. HELPER: GEOMETRIC MASK GENERATOR
# ==========================================
def generate_sparse_mask(coords, radius):
    """
    Creates an Attention Mask based on 3D Euclidean distance.
    Returns:
        mask: [N, N] tensor where True (or -inf) means "IGNORE THIS CONNECTION"
    """
    # coords shape: [N, 3]
    # Calculate Pairwise Distance Matrix: [N, N]
    dist_matrix = torch.cdist(coords, coords, p=2)
    
    # Create Mask: True where distance > radius (Too far to care)
    # We allow self-attention (dist=0) always.
    mask = dist_matrix > radius
    
    return mask

# ==========================================
# 4. SPARSE VISION TRANSFORMER
# ==========================================
class ParticleViT(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Standard Transformer Encoder
        # We will inject the mask during the forward pass
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 8) 
        )

    def forward(self, x, mask=None):
        # x: [Batch, N, 3]
        # mask: [N, N] Boolean mask (True = Ignore)
        
        emb = self.embedding(x) + self.pos_encoder(x)
        
        # Pass the mask to the transformer
        # Note: PyTorch Transformer expects mask shape (N, N) or (Batch*NumHeads, N, N)
        # We assume Batch=1 for this implementation.
        feat = self.transformer(emb, mask=mask)
        
        out = self.head(feat)
        return out

# ==========================================
# 5. CONTRASTIVE LOSS (Unchanged)
# ==========================================
def contrastive_loss(embeddings, labels, margin=1.0):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.unsqueeze(0)
    mask = torch.eq(labels, labels.T).float()
    
    pos_loss = mask * torch.pow(dist_matrix, 2)
    neg_loss = (1 - mask) * torch.pow(torch.clamp(margin - dist_matrix, min=0.0), 2)
    
    loss = torch.sum(pos_loss) + torch.sum(neg_loss)
    loss /= (embeddings.shape[0] * embeddings.shape[0])
    return loss

# ==========================================
# 6. SCORE METRIC (Unchanged)
# ==========================================
def score_double_majority(pred_labels, true_labels, weights):
    unique_preds = np.unique(pred_labels)
    total_score = 0.0
    df_temp = pd.DataFrame({'pred': pred_labels, 'truth': true_labels, 'weight': weights})
    
    for pid in unique_preds:
        if pid == -1: continue 
        track_hits = df_temp[df_temp['pred'] == pid]
        if len(track_hits) == 0: continue
        real_hits = track_hits[track_hits['truth'] != 0]
        if len(real_hits) == 0: continue
        majority_particle = real_hits['truth'].mode()[0]
        n_match = len(track_hits[track_hits['truth'] == majority_particle])
        n_pred = len(track_hits)
        n_truth_total = len(df_temp[df_temp['truth'] == majority_particle])
        if n_match > 0.5 * n_pred and n_match > 0.5 * n_truth_total:
            total_score += track_hits[track_hits['truth'] == majority_particle]['weight'].sum()
    return total_score

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    # --- Load Data ---
    train_ds = ParticleDataset("simple_dataset_clean1.xlsx", is_train=True)
    test_ds = ParticleDataset("medium_dataset_noisy1.xlsx", scaler=train_ds.scaler, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model = ParticleViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n⚡ Training Sparse ViT (Radius={SPARSE_RADIUS}) for {EPOCHS} Epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            
            # --- GENERATE SPARSE MASK ---
            # We calculate distance on the normalized coords x
            # Since batch size is 1, we squeeze to get [N, 3]
            coords = x.squeeze(0)
            sparse_mask = generate_sparse_mask(coords, radius=SPARSE_RADIUS).to(device)
            
            optimizer.zero_grad()
            
            # Pass mask to model
            embeddings = model(x, mask=sparse_mask) 
            
            loss = contrastive_loss(embeddings.squeeze(0), y.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # --- Testing ---
    print("\n🔍 Running Inference (Sparse Mode)...")
    model.eval()
    final_scores = []
    
    with torch.no_grad():
        for x, y, w, ev_id in tqdm(test_loader):
            x = x.to(device)
            
            # Generate Mask for Test Data too
            coords = x.squeeze(0)
            sparse_mask = generate_sparse_mask(coords, radius=SPARSE_RADIUS).to(device)
            
            # Inference
            embeddings = model(x, mask=sparse_mask).squeeze(0).cpu().numpy()
            
            # Clustering
            clustering = DBSCAN(eps=0.2, min_samples=3).fit(embeddings)
            pred_labels = clustering.labels_
            
            score = score_double_majority(pred_labels, y.squeeze(0).numpy(), w.squeeze(0).numpy())
            final_scores.append(score)
            
            if len(final_scores) == 1:
                n_tracks = len(np.unique(pred_labels)) - (1 if -1 in pred_labels else 0)
                print(f"\n[Event {ev_id.item()}] Found {n_tracks} tracks. Score: {score:.4f}")

    print("\n" + "="*40)
    print(f"🏆 FINAL SCORE (SPARSE ATTENTION): {np.mean(final_scores):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
