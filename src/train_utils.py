import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    loop = tqdm(dataloader, desc="Training", leave=False)
    for wavs, labels in loop:
        wavs, labels = wavs.to(device), labels.to(device).float().unsqueeze(1)

        logits = model(wavs)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).int()
        y_true.extend(labels.cpu().numpy().astype(int))
        y_pred.extend(preds.cpu().numpy())
        
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred)
    
    return avg_loss, acc


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0

    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for wavs, labels in loop:
            wavs, labels = wavs.to(device), labels.to(device).float().unsqueeze(1)

            logits = model(wavs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            y_true.extend(labels.cpu().numpy().astype(int))
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred)
    roc_auc_classwise = roc_auc_score(y_true, y_pred, average=None)
    
    return avg_loss, acc, roc_auc_classwise