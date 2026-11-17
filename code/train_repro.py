"""Reproduction of ResNet-TCN (abedidev/ResNet-TCN) - FAITHFUL to the repo's train.py.

Kept unchanged: ResTCN architecture, SGD lr .001 momentum .9, batch 4, 100 epochs,
torch.manual_seed(0), CE, no scheduler step, Test evaluated every epoch (as in the repo -
that is their protocol). Only differences: (a) our csv/paths,
(b) jsonl log written every epoch, (c) Test probabilities saved at the best-acc epoch and the
last epoch (to run Audit 2 on THEIR predictions), (d) num_workers for the dataloader.

Run: ~/p2/venv/bin/python train_repro.py  (cwd = ~/p2/repro/ResNet-TCN)
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

# the repo's utils.get_dataloader wraps VideoDataset + transform; rebuilt directly here:
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
