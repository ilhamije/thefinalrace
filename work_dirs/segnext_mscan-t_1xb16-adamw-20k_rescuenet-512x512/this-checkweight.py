
import os
import torch


def load_weights(checkpoint_path, layer_name=None):
    """Load weights from the checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', {})

    if layer_name:
        # Return weights of the specific layer
        if layer_name in state_dict:
            return state_dict[layer_name]
        else:
            raise ValueError(
                f"Layer '{layer_name}' not found in the checkpoint.")
    else:
        # Return the entire state_dict
        return state_dict


def get_work_dirs(base_path):
    """Dynamically get all work directories."""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


def process_work_dirs(layer_name):
    """Process each work directory to load weights."""
    work_dirs = get_work_dirs(base_path)
    for config in work_dirs:
        # Assuming 'latest.pth' is the checkpoint
        checkpoint_path = os.path.join(base_path, config, '*.pth')
        try:
            weights = load_weights(checkpoint_path, layer_name=layer_name)
            print(f"work_dirs/{config} : {weights}")
        except Exception as e:
            print(f"work_dirs/{config} : Error - {str(e)}")


layer_name = "backbone.conv1.weight"

# Run the process
process_work_dirs(layer_name)

