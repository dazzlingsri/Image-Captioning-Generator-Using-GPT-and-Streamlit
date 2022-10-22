import io
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, scaler, optimizer, loader, epoch, device='cpu'):
    model.train()

    total_loss = 0

    loop = tqdm(loader, total=len(loader))
    loop.set_description(f'Epoch: {epoch} | Loss: ---')
    for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

        img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

        with torch.cuda.amp.autocast():
            loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        total_loss += loss.item()

        loop.set_description(f'Epoch: {epoch} | Loss: {total_loss / (batch_idx + 1):.3f}')
        loop.refresh()

    return {
        'loss': total_loss / (batch_idx + 1)
    }

def valid_epoch(model, loader, device='cpu'):
    model.eval()

    total_loss = 0

    loop = tqdm(loader, total=len(loader))
    loop.set_description(f'Validation Loss: ---')
    for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

        img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():

                loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)

                total_loss += loss.item()
                
                loop.set_description(f'Validation Loss: {total_loss / (batch_idx + 1):.3f}')
                loop.refresh()

    return {
        'loss': total_loss / (batch_idx + 1)
    }

def test_step(model, dataset, img_path, num_examples=4):
    assert num_examples % 2 == 0, 'num_examples must be even'

    model.eval()

    fig, axs = plt.subplots(num_examples // 2, 2, figsize=(20, 12))

    random_idx = np.random.randint(0, len(dataset), size=(num_examples,))
    for idx, r in enumerate(random_idx):
        img_name, _, _ = dataset[r]

        img = Image.open(os.path.join(img_path, img_name))

        with torch.no_grad():
            caption, _ = model(img)

        axs[idx // 2, idx % 2].imshow(img)
        axs[idx // 2, idx % 2].set_title(caption)
        axs[idx // 2, idx % 2].axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    fig.clear()
    plt.close(fig)

    return Image.open(buf)