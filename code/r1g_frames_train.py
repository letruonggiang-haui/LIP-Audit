"""R1 registration, Section 4G: ResNet-TCN frame ablation on DAiSEE validation.

The released training recipe of abedidev/ResNet-TCN, unchanged (ResTCN,
ImageNet-pretrained ResNet-18, SGD lr 1e-3 momentum 0.9, batch 4, 100 epochs,
torch.manual_seed(0), unweighted cross-entropy, the repository's frame
transform with its 80:560 column crop and 224 resize), except:
  - the number of time steps follows the arm (16 or 50) instead of the fixed 16;
  - it trains on the training split only and reads the VALIDATION split after
    every epoch (the released script reads the test split; no test file exists
    for this ablation);
  - the validation probabilities of every epoch are saved.

Run: ~/p2/venv/bin/python r1g_frames_train.py --arm 50u
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from torch import nn, optim

sys.path.insert(0, os.path.expanduser("~/p2/repro/ResNet-TCN"))
from ResTCN import ResTCN  # noqa: E402
from datasets import VideoDataset  # noqa: E402

R1 = os.path.expanduser("~/r1/g")


class FolderToTensor(object):
    """transforms.VideoFolderPathToTensor of the repository with num_time_steps
    as a parameter (EXTRACT_FREQUENCY = 1, as released)."""

    def __init__(self, num_time_steps):
        self.num_time_steps = num_time_steps
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Resize([224, 224])
        ])

    def __call__(self, path):
        file_names = sorted([os.path.join(path, f) for f in os.listdir(path)
                             if os.path.isfile(os.path.join(path, f))])
        frame = cv2.imread(file_names[0])
        height, width, channels = frame.shape
        frames = torch.FloatTensor(channels, self.num_time_steps, 224, 224)
        for index in range(0, self.num_time_steps):
            frame = cv2.imread(file_names[index])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frame = frame / 255
            if frame.shape[2] != 224:
                frame = frame[:, :, 80:560]
            frame = self.transform(frame)
            frames[:, index, :, :] = frame.float()
        return frames.permute(1, 0, 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["16c", "16u", "50u"])
    ap.add_argument("--epochs", type=int, default=100)
    # Added 27/10/2026 (DEVIATIONS_R1.md, entry 6): extra seeds for A50u. Seed 0 keeps the
    # registered run directory; another seed writes run_<arm>_s<seed>.
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    steps = 50 if args.arm == "50u" else 16
    csv_tr = os.path.join(R1, "%s_train.csv" % args.arm)
    csv_va = os.path.join(R1, "%s_validation.csv" % args.arm)
    for c in (csv_tr, csv_va):
        assert "test" not in c.lower(), c
    out = os.path.join(R1, "run_" + args.arm + ("" if args.seed == 0 else "_s%d" % args.seed))
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tf = FolderToTensor(steps)
    train_ds = VideoDataset(csv_tr, "", transform=tf)
    val_ds = VideoDataset(csv_va, "", transform=tf)
    for ds in (train_ds, val_ds):
        assert not ds.dataframe["path"].str.contains("/Test/").any()
    dl_train = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True,
                                           num_workers=12, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False,
                                         num_workers=12, pin_memory=True)
    ids = np.array([os.path.basename(p) for p in val_ds.dataframe["path"]])
    yv = val_ds.dataframe["label"].values.astype(int)
    print({"arm": args.arm, "steps": steps, "train": len(train_ds), "val": len(val_ds)}, flush=True)

    model = ResTCN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim=1)
    log = open(os.path.join(out, "log.jsonl"), "a")
    all_p = []
    for epoch in range(args.epochs):
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
        probs = []
        with torch.no_grad():
            for inputs, _ in dl_val:
                outputs = model(inputs.to(device, non_blocking=True))
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                probs.append(softmax(outputs).cpu().numpy())
        P = np.concatenate(probs).astype(np.float32)
        all_p.append(P)
        rec = dict(ep=epoch, loss=running / len(train_ds), val_acc=float((P.argmax(1) == yv).mean()),
                   mins=round((time.time() - t0) / 60, 1))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        np.savez_compressed(os.path.join(out, "val_probs_all_epochs.npz"),
                            P=np.stack(all_p), ids=ids, y=yv)
        # Stop rule of DEVIATIONS_R1.md entry 6, for the added seeds only.
        if args.seed != 0 and epoch == 0 and float(P.std(0).max()) < 1e-4:
            open(os.path.join(out, "COLLAPSED"), "w").write(json.dumps({"epoch": 0, "max_std": float(P.std(0).max())}))
            print("collapsed after epoch 0; stopped", flush=True)
            return
    open(os.path.join(out, "DONE"), "w").close()


if __name__ == "__main__":
    main()
