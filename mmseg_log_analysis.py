
import json
import os
import re
import matplotlib.pyplot as plt

def load_log(log_path):
    logs = []
    if log_path.endswith('.json'):
        # Handle line-by-line JSON logs
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except:
                    continue
    else:
        # Plain text logs
        with open(log_path, 'r') as f:
            logs = f.readlines()
    return logs

def extract_losses(logs, mode='train'):
    iters, losses = [], []
    for entry in logs:
        if isinstance(entry, dict):
            if entry.get('mode') == mode and 'loss' in entry:
                iters.append(entry['iter'])
                losses.append(entry['loss'])
        elif isinstance(entry, str):
            match = re.search(rf"Iter\({mode}\)\s+\[\s*(\d+)/\d+\].*?loss:\s+([\d.]+)", entry)
            if match:
                iters.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return iters, losses

def extract_miou(logs, iters_per_epoch=2000):
    miou_vals, epochs = [], []
    current_epoch = 1
    for entry in logs:
        if isinstance(entry, dict):
            if 'mIoU' in entry:
                miou_vals.append(entry['mIoU'])
                epochs.append(current_epoch)
                current_epoch += 1
        elif isinstance(entry, str):
            match = re.search(r"mIoU:\s+([\d.]+)", entry)
            if match:
                miou_vals.append(float(match.group(1)))
                epochs.append(current_epoch)
                current_epoch += 1
    return epochs, miou_vals

def plot_loss(train_iters, train_loss, val_iters=None, val_loss=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_iters, train_loss, label='Training Loss', color='crimson', linewidth=2)
    if val_iters and val_loss:
        ax.plot(val_iters, val_loss, label='Validation Loss', color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_miou(models_logs, iters_per_epoch=2000):
    plt.figure(figsize=(10, 5))
    for label, logs in models_logs.items():
        epochs, miou_vals = extract_miou(logs, iters_per_epoch)
        plt.plot(epochs, miou_vals, marker='o', linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# logs1 = load_log("work_dirs/model1/log.json")
# logs2 = load_log("work_dirs/model2/log.json")
# plot_miou({"Model A": logs1, "Model B": logs2})
