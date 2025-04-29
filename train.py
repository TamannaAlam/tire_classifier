import torch, torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# ---------- Config ----------
DATA_DIR = Path("./")         # project root
BATCH_SIZE = 16               # lower if you run out of RAM/VRAM
EPOCHS = 15
LR = 3e-4
IMG_SIZE = 224                # EfficientNet-B0 native
NUM_WORKERS = 0               # set 0 on Windows if you hit bugs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------

train_dir = DATA_DIR / "train"
val_dir   = DATA_DIR / "val"

# --- 1. Data pipeline ---
train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
])

val_tfms = transforms.Compose([
        transforms.Resize(int(IMG_SIZE*1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
])

train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = torchvision.datasets.ImageFolder(val_dir,   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- 2. Model ---
model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0
for epoch in range(1, EPOCHS+1):
    # ---- Training ----
    model.train()
    total_loss = 0
    for x,y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=80):
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    scheduler.step()

    # ---- Validation ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    val_acc = correct / total * 100
    print(f"  train-loss {total_loss/len(train_ds):.4f} | val-acc {val_acc:.2f}%")

    # ---- Save best ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_effnet_tires.pt")
        print("  ✅ Saved new best model")

print("Done! Best validation accuracy:", best_acc)
