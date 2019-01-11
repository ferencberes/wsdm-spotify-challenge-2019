import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import sklearn.metrics
import sys


split_folder = sys.argv[1]

print("reading")
combined_fixed_length = np.load(split_folder + "/train/combined_fixed_length_orig.npy", )
max_track = max(-combined_fixed_length.min(), combined_fixed_length.max())

split_on = ((len(combined_fixed_length) - (len(combined_fixed_length) // 20)) // 20) * 20
train_data = combined_fixed_length[:split_on]
eval_data = combined_fixed_length[split_on:]

print("shuffling")
train_data = train_data.reshape(-1, 20)
np.random.shuffle(train_data)
train_data = train_data.reshape(-1)


def data_iter(data, f, ll, bsize, round_batch=False):
    fbsize = bsize * 20
    alternating = np.zeros(bsize * 2)
    alternating[::2] = 1

    findex = f * ll * 20
    if round_batch:
        tindex = min(len(data) - len(data) % fbsize, (f + 1) * ll * 20)
    else:
        tindex = min(len(data), (f + 1) * ll * 20)

        leftover = (len(data) // 20) % bsize
        alternating_last = np.zeros(leftover * 2)
        alternating_last[::2] = 1

    for i in range(findex, tindex, fbsize):
        batch = data[i:i + fbsize].reshape(-1, 20)
        items = np.abs(batch).astype(np.int64)
        targets = np.sign(batch).astype(np.float32)
        mask_l = (targets != 0).sum(axis=1) // 2
        mask_cl = 20 - mask_l
        if batch.shape[0] == bsize:
            mask = np.repeat(alternating, np.column_stack([mask_l, mask_cl]).ravel()).reshape(-1, 20).astype(np.int64)
            yield (items, targets, mask)
        else:
            mask = np.repeat(alternating_last, np.column_stack([mask_l, mask_cl]).ravel()).reshape(-1, 20).astype(np.int64)
            yield (items, targets, mask)


def data_iter_notrainmask(data, f, ll, bsize, round_batch=False):
    fbsize = bsize * 20
    alternating = np.zeros(bsize * 2)
    alternating[::2] = 1

    findex = f * ll * 20
    if round_batch:
        tindex = min(len(data) - len(data) % fbsize, (f + 1) * ll * 20)
    else:
        tindex = min(len(data), (f + 1) * ll * 20)

        leftover = (len(data) // 20) % bsize
        alternating_last = np.zeros(leftover * 2)
        alternating_last[::2] = 1

    for i in range(findex, tindex, fbsize):
        batch = data[i:i + fbsize].reshape(-1, 20)
        items = np.abs(batch).astype(np.int64)
        targets = np.sign(batch).astype(np.float32)
        mask_l = (targets != 0).sum(axis=1)
        mask_cl = 20 - mask_l
        if batch.shape[0] == bsize:
            mask = np.repeat(alternating, np.column_stack([mask_l, mask_cl]).ravel()).reshape(-1, 20).astype(np.int64)
            yield (items, targets, mask)
        else:
            mask = np.repeat(alternating_last, np.column_stack([mask_l, mask_cl]).ravel()).reshape(-1, 20).astype(np.int64)
            yield (items, targets, mask)


print("allocating cuda resources")

device = "cuda:0"
torch.backends.cudnn.benchmark = True

encoder = nn.Embedding(int(max_track + 1), embedding_dim=128, sparse=True, padding_idx=torch.LongTensor([0])[0]).to(device)
decoder = nn.Embedding(int(max_track + 1), embedding_dim=128, sparse=True, padding_idx=torch.LongTensor([0])[0]).to(device)

encoder.weight.data = encoder.weight.data / 128
decoder.weight.data = decoder.weight.data * 0


optimizer = optim.Adagrad(
    list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

losses = []


def train(j, evall=False):
    iterator = data_iter_notrainmask(train_data, j, 5000000, 400, round_batch=True)
    for (items, targets, train_mask) in iterator:
        items_torch = torch.from_numpy(items).to(device)
        mask = torch.from_numpy(train_mask).to(device)
        targets_torch = torch.from_numpy(targets).to(device)
        randhide = (torch.rand(items_torch.shape) >= 0.5).type(torch.LongTensor).to(device)

        in_embeds = F.dropout(encoder(items_torch * mask * randhide), 0.3)
        out_embeds = decoder(items_torch)
        h1 = (in_embeds * targets_torch.unsqueeze(2)).sum(1)
        logits = torch.matmul(out_embeds, h1.unsqueeze(2)).squeeze()
        loss = (logits - (targets_torch)).pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    if evall:
        print("evaling")
        aucs = []
        iterator = data_iter(eval_data, 0, 5000000, 400, round_batch=True)
        for (items, targets, train_mask) in iterator:
            with torch.no_grad():
                items_torch = torch.from_numpy(items).to(device)
                mask = torch.from_numpy(train_mask).to(device)
                targets_torch = torch.from_numpy(targets).to(device)

                in_embeds = encoder(items_torch * mask)
                out_embeds = decoder(items_torch)
                h1 = (in_embeds * (targets_torch).unsqueeze(2)).sum(1)
                logits = torch.matmul(out_embeds, h1.unsqueeze(2)).squeeze()
                scores_np = logits.detach().cpu().numpy().reshape(-1)

                test_mask = (items != 0) & (1 - train_mask)
                targets_ = (targets.reshape(-1)[test_mask.reshape(-1) != 0] + 1) / 2
                scores_ = scores_np[test_mask.reshape(-1) != 0]
                if(not (np.all(1 - targets_) or np.all(targets_))):
                    sc = sklearn.metrics.roc_auc_score(targets_, scores_)
                aucs.append(sc)
        print(np.mean(aucs))


for j in range(19):
    print("run", j)
    train(j, True)

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
}, split_folder + "/models/autoencoder_1")
