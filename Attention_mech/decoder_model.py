import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import GPT2Config, GPT2Model


class GPT2Regressor(nn.Module):
    def __init__(self, seq_length=20, hidden_size=128, n_layers=2, n_heads=4, dropout=0.1):
        super(GPT2Regressor, self).__init__()
        self.seq_length = seq_length
        
        config = GPT2Config(
            vocab_size=1,        
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=hidden_size,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        
        self.input_embedding = nn.Linear(1, hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_size))
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embedding(x) 
        x = x + self.pos_embedding[:, :self.seq_length, :]  
        transformer_outputs = self.transformer(inputs_embeds=x)
        hidden_states = transformer_outputs.last_hidden_state 
        pooled_output = hidden_states.mean(dim=1)  
        pooled_output = self.dropout(pooled_output)
        output = self.regressor(pooled_output)  
        return output
    

def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics: MSE, RMSE, MAE, and R².
    
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        dict: Dictionary with metrics.
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def main():
    # Load the dataset.
    df = pd.read_csv("data/TASK-ML-INTERN.csv")
    
    df = df.drop(df.columns[0], axis=1)
    
    X = df.iloc[:, :20].values.astype(np.float32)
    y = df["vomitoxin_ppb"].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1) 
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Regressor(seq_length=20, hidden_size=128, n_layers=2, n_heads=4, dropout=0.1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            all_preds.append(preds)
            all_targets.append(batch_y)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_targets, all_preds)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()