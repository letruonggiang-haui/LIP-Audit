"""Tai hien ResNet-TCN (abedidev/ResNet-TCN) — TRUNG THANH voi train.py cua repo.

Giu nguyen: kien truc ResTCN, SGD lr .001 momentum .9, batch 4, 100 epoch,
torch.manual_seed(0), CE, khong scheduler step, danh gia Test moi epoch (nhu repo —
day chinh la giao thuc cua ho). Khac duy nhat: (a) csv/duong dan cua ta,
(b) ghi log jsonl moi epoch, (c) luu xac suat Test tai epoch best-acc va epoch cuoi
(de chay Audit 2 tren prediction cua HO), (d) num_workers cho dataloader.

Chay: ~/p2/venv/bin/python train_repro.py  (cwd = ~/p2/repro/ResNet-TCN)
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.expanduser("~/p2/repro/ResNet-TCN"))
from ResTCN import ResTCN  # noqa: E402
from datasets import VideoDataset  # noqa: E402
from transforms import VideoFolderPathToTensor  # noqa: E402

torch.manual_seed(0)
num_epochs = 100
batch_size = 4
lr = .001
device = torch.device("cuda")
OUT = os.path.expanduser("~/p2/repro")

# repo utils.get_dataloader boc VideoDataset + transform; tai tao truc tiep:
transform = VideoFolderPathToTensor()
train_ds = VideoDataset(os.path.join(OUT, "train_repro.csv"), "", transform=transform)
test_ds = VideoDataset(os.path.join(OUT, "test_repro.csv"), "", transform=transform)
dl_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=10, pin_memory=True)
dl_test = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=10, pin_memory=True)
print({"train": len(train_ds), "test": len(test_ds)}, flush=True)

model = ResTCN().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)
softmax = nn.Softmax(dim=1)

log = open(os.path.join(OUT, "repro_log.jsonl"), "a")
best_acc = -1.0
for epoch in range(num_epochs):
    t0 = time.time()
    model.train()
    running = 0.0
    for inputs, labels in dl_train:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.long().to(device)
        outputs = model(inputs)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * inputs.size(0)

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for inputs, labels in dl_test:
            outputs = model(inputs.to(device, non_blocking=True))
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            probs.append(softmax(outputs).cpu().numpy())
            trues.append(labels.numpy())
    P = np.concatenate(probs)
    Y = np.concatenate(trues).astype(int)
    acc = accuracy_score(Y, P.argmax(1))
    rec = dict(ep=epoch, loss=running / len(train_ds), test_acc=float(acc),
               mins=round((time.time() - t0) / 60, 1))
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    np.savez(os.path.join(OUT, "repro_test_probs_last.npz"), P=P, y=Y, ep=epoch)
    if acc > best_acc:
        best_acc = acc
        np.savez(os.path.join(OUT, "repro_test_probs_best.npz"), P=P, y=Y, ep=epoch)
print(f"DONE best_test_acc={best_acc:.4f}", flush=True)
