import os
import py3nvml
import yaml

# Load configuration from the YAML file
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    try:
        # Specify the configuration file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../config.yaml")  # Relative to the script's location
        print(f'--> Config file path read as {config_path}')
        config = load_config(config_path)

        # Extract values from the configuration
        gpu_resources = config['jobs'].get('gpu_resources', 'gpu:1')
        gpu_fraction = float(config['jobs'].get('gpu_usage_threshold', 0.95))

        # Validate GPU settings
        if not gpu_resources.startswith('gpu:') or not gpu_resources[-1].isdigit():
            raise ValueError(f"Invalid gpu_resources format: {gpu_resources}")
        if not (0.0 < gpu_fraction <= 1.0):
            raise ValueError(f"Invalid gpu_usage_threshold: {gpu_fraction}")

        num_gpus = int(gpu_resources[-1])  # Extract the number of GPUs
        print(f"Requesting {num_gpus} GPU(s) with at least {gpu_fraction * 100:.2f}% memory free.")

        # Grab GPUs dynamically
        py3nvml.grab_gpus(num_gpus=num_gpus, gpu_fraction=gpu_fraction)

        # Confirm GPU allocation
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        try:
            os.system("nvidia-smi")
        except OSError as e:
            print(f"nvidia-smi not available: {e}")
    except Exception as e:
        print(f"Failed to grab GPUs: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Fallback to CPU if no GPUs are available

if __name__ == "__main__":
    main()
