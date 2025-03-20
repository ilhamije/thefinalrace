import json
import time
import torch
import subprocess
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets import build_dataset

# Configuration
CONFIG_PATH = "configs/your_model_config.py"
CHECKPOINT_PATH = "work_dirs/your_model/latest.pth"
LOG_PATH = "work_dirs/your_model/log.json"

# Load Model
model = init_segmentor(CONFIG_PATH, CHECKPOINT_PATH, device='cuda:0')


def get_training_time():
    """Extract training time per epoch from log.json."""
    with open(LOG_PATH, 'r') as f:
        logs = [json.loads(line) for line in f]
    epoch_times = [log['time'] for log in logs if 'time' in log]
    avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    return avg_time


def get_inference_time():
    """Measure average inference time per image."""
    dataset = build_dataset(model.cfg.data.test)
    num_samples = len(dataset)
    total_time = 0
    for i in range(num_samples):
        img = dataset[i]['img']
        start_time = time.time()
        with torch.no_grad():
            _ = inference_segmentor(model, img)
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time / num_samples if num_samples else 0


def get_gpu_memory_usage():
    """Check GPU memory usage using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True)
    memory_used = result.stdout.strip().split(
        '\n')[0]  # Get first GPU memory value
    return float(memory_used) / 1024  # Convert MB to GB


def get_flops_and_params():
    """Run get_flops.py to measure FLOPs and model parameters."""
    result = subprocess.run(["python", "tools/get_flops.py", CONFIG_PATH, "--shape", "3", "512", "512"],
                            capture_output=True, text=True)
    lines = result.stdout.split('\n')
    flops, params = None, None
    for line in lines:
        if "FLOPs:" in line:
            flops = float(line.split(':')[1].strip().split()[
                          0])  # Extract FLOPs value
        if "Params:" in line:
            params = float(line.split(':')[1].strip().split()[
                           0])  # Extract Parameters value
    return flops, params


def compute_ces(miou, train_time, infer_time):
    """Compute Compute Efficiency Score (CES)."""
    return miou / (train_time * infer_time) if train_time and infer_time else 0


if __name__ == "__main__":
    train_time = get_training_time()
    infer_time = get_inference_time()
    gpu_memory = get_gpu_memory_usage()
    flops, params = get_flops_and_params()

    # Assuming mIoU is stored in log.json (modify if needed)
    with open(LOG_PATH, 'r') as f:
        logs = [json.loads(line) for line in f]
    miou = max([log['mIoU'] for log in logs if 'mIoU' in log], default=0)

    ces = compute_ces(miou, train_time, infer_time)

    print(f"Training Time per Epoch: {train_time:.2f} hours")
    print(f"Inference Time per Image: {infer_time:.4f} seconds")
    print(f"GPU Memory Usage: {gpu_memory:.2f} GB")
    print(f"FLOPs: {flops} GFLOPs")
    print(f"Parameters: {params} Million")
    print(f"Compute Efficiency Score (CES): {ces:.2f}")
