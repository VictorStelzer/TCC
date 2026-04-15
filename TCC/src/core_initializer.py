import torch

def validate_hardware():
    print("Iniciando validação de hardware...")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[OK] GPU identificada e pronta para processamento: {device_name}")
        return "cuda"
    else:
        print("[WARNING] CUDA não disponível. O processamento será realizado na CPU.")
        return "cpu"

if __name__ == "__main__":
    validate_hardware()
