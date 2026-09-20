"""
train_mlp.py -- Training MLP: standalone, no integration with sklearn pipeline.
Run: python src/train_mlp.py
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config as config
from data import load_train, load_test
from features import data_processing


# ============================================================================
# MLP -- current architecture (2 layers, unchanged)
# ============================================================================
class MLP(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# MLPTrainer -- wrapper: preprocessing + training + inference
# ============================================================================
class MLPTrainer:
    """
    Full MLP cycle:
      - StandardScaler for numeric features
      - OneHotEncoder for categorical features
      - fit / predict / predict_proba
    """

    def __init__(self, hidden=64, lr=1e-3, epochs=100, batch_size=64,
                 device='cpu', random_state=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        # Preprocessing column lists
        self.numeric_cols = config.NUMERIC_COLS
        self.categorical_cols = config.CATEGORY_COLS
        self.n_features = None
        self.model = None
        self.scaler = None
        self.encoder = None

    def _preprocess(self, df: pd.DataFrame, fit=False):
        """
        Convert DataFrame to torch tensor.
        fit=True -> fit scaler + encoder, remember n_features.
        """
        if fit:
            self.scaler = StandardScaler()
            self.encoder = OneHotEncoder(sparse_output=False,
                                          handle_unknown='ignore')

            num_data = self.scaler.fit_transform(df[self.numeric_cols].fillna(0))
            cat_data = self.encoder.fit_transform(df[self.categorical_cols].fillna('NA'))
            combined = np.hstack([num_data, cat_data])

            self.n_features = combined.shape[1]
        else:
            num_data = self.scaler.transform(df[self.numeric_cols].fillna(0))
            cat_data = self.encoder.transform(df[self.categorical_cols].fillna('NA'))
            combined = np.hstack([num_data, cat_data])

        return torch.tensor(combined, dtype=torch.float32, device=self.device)

    def fit(self, train_df: pd.DataFrame, y: np.ndarray):
        """Train the model on a full dataset."""
        X = self._preprocess(train_df, fit=True)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.model = MLP(self.n_features, self.hidden).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X, y_tensor)
        loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = total_loss / n_batches
                print(f'  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """predict: 0 or 1."""
        self.model.eval()
        X = self._preprocess(df, fit=False)
        with torch.no_grad():
            logits = self.model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        return (probs > 0.5).astype(bool)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """predict_proba: probability of class 1."""
        self.model.eval()
        X = self._preprocess(df, fit=False)
        with torch.no_grad():
            logits = self.model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ============================================================================
# CV
# ============================================================================
def evaluate_mlp_cv(train_df, hidden=64, lr=1e-3, epochs=100,
                    batch_size=64, device='cpu', n_splits=5):
    """
    StratifiedKFold CV.
    On each fold: fit scaler+encoder on train fold, evaluate on val fold.
    """
    X = train_df.drop(columns=['Transported'])
    y = train_df['Transported'].astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_STATE)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\n--- Fold {fold + 1}/{n_splits} ---')
        t0 = time.time()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        trainer = MLPTrainer(hidden=hidden, lr=lr, epochs=epochs,
                             batch_size=batch_size, device=device)
        trainer.fit(X_train, y_train)

        preds = trainer.predict(X_val)
        acc = (preds == y_val).mean()
        elapsed = time.time() - t0
        print(f'  Accuracy: {acc:.4f}  [{elapsed:.1f}s]')
        scores.append(acc)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f'\n=== CV Result: {mean_acc:.4f} +/- {std_acc:.4f} ===')
    return mean_acc, std_acc, scores


# ============================================================================
# Full cycle
# ============================================================================
def run_mlp(hidden=64, lr=1e-3, epochs=100, batch_size=64,
            device='cpu', n_splits=5, output_path='submissions/submission_mlp.csv'):
    """
    Load -> CV -> train on all -> submission.
    """
    print('=== Loading data ===')
    train_raw = load_train()
    test_raw = load_test()
    test_ids = test_raw['PassengerId'].copy()

    print('=== Processing data ===')
    train = data_processing(train_raw)
    test = data_processing(test_raw)

    # CV
    print('\n=== CV ===')
    mean_acc, std_acc, _ = evaluate_mlp_cv(
        train, hidden=hidden, lr=lr, epochs=epochs,
        batch_size=batch_size, device=device, n_splits=n_splits,
    )

    # Final training on all train data
    print('\n=== Final training on all train ===')
    X_train = train.drop(columns=['Transported'])
    y_train = train['Transported'].astype(int).values

    final_trainer = MLPTrainer(hidden=hidden, lr=lr, epochs=epochs,
                                batch_size=batch_size, device=device)
    final_trainer.fit(X_train, y_train)

    # Submission
    print('\n=== Submission ===')
    preds = final_trainer.predict(test)
    probs = final_trainer.predict_proba(test)

    sub = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': preds,
    })
    sub.to_csv(output_path, index=False)

    print(f'Share of True: {preds.mean():.4f}')
    print(f'x {output_path}: {sub.shape}')
    return sub


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' # мало данных, на CPU быстрее
    print(f'Using device: {device}')
    run_mlp(device=device, n_splits=5)
