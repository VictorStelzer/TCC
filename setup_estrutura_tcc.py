import os
import yaml

def create_directory_structure(base_dir):
    """Cria a estrutura de pastas principal e a hierarquia do YOLOv8."""
    directories = [
        "data/raw_dataset/images/train",
        "data/raw_dataset/images/val",
        "data/raw_dataset/labels/train",
        "data/raw_dataset/labels/val",
        "models",
        "src",
        "scripts"
    ]
    
    for relative_path in directories:
        path = os.path.join(base_dir, relative_path)
        os.makedirs(path, exist_ok=True)

def create_environment_config(base_dir):
    """Gera o arquivo de configurações de ambiente e classes."""
    config = {
        "classes": [
            "pedestre",
            "veiculo",
            "calcada",
            "degrau",
            "obstaculo_suspenso",
            "faixa_pedestre"
        ]
    }
    config_path = os.path.join(base_dir, "environment_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def create_core_initializer(base_dir):
    """Cria o script python para validar se a GPU ou CPU está pronta."""
    content = """import torch

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
"""
    path = os.path.join(base_dir, "src", "core_initializer.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def create_trainer_engine(base_dir):
    """Cria o código com o boilerplate do YOLOv8 Nano."""
    content = """from ultralytics import YOLO

def load_yolo_model():
    print("Carregando o modelo YOLOv8 Nano...")
    # Inicializa o modelo YOLO usando os pesos nano padrão
    model = YOLO("yolov8n.pt")
    print("[OK] Modelo carregado com sucesso.")
    return model

if __name__ == "__main__":
    model = load_yolo_model()
"""
    path = os.path.join(base_dir, "src", "trainer_engine.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def create_stream_validator(base_dir):
    """Cria um script simples para testar o OpenCV e fluxo de vídeo."""
    content = """import cv2

def validate_video_stream(camera_index=0):
    print(f"Iniciando captura de vídeo na fonte {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível acessar a câmera {camera_index}.")
        return
        
    print("[OK] Fluxo de vídeo capturado. Pressione 'q' na janela para encerrar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame atual.")
            break
            
        cv2.imshow('Core: Validador de Stream', frame)
        
        # Aguarda tecla q para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Encerrando stream...")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    validate_video_stream()
"""
    path = os.path.join(base_dir, "src", "stream_validator.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def create_readme(base_dir):
    """Cria o README inicial do projeto."""
    content = """# Core

Sistema de visão computacional profissional focado em inferência on-device para auxílio na navegação.

## Estrutura do Projeto

* `data/`: Dados brutos e processados para o treinamento do YOLOv8.
* `models/`: Armazena os pesos do modelo exportados (`.pt`, `.tflite`).
* `src/`: Core do projeto (código-fonte principal).
  * `core_initializer.py`: Validação se o hardware (GPU/CPU) está pronto.
  * `trainer_engine.py`: Boilerplate para carregamento e treinamento de modelos (YOLOv8 Nano).
  * `stream_validator.py`: Validação de rotinas do OpenCV e fluxo de captura.
* `scripts/`: Utilitários gerais e scripts de conversão on-device.
* `environment_config.yaml`: Metadados e classes mapeadas (pedestre, veiculo, entre outros).
"""
    path = os.path.join(base_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def print_directory_tree(startpath):
    """Imprime uma árvore de diretórios simulando o comando 'tree'."""
    print(f"\\n> Árvore de Diretórios: {startpath}")
    print("=" * 40)
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}📂 {os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}📄 {f}')
    print("=" * 40)

def main():
    root_folder = "TCC"
    print("Iniciando o bootstrap do Core...")
    
    # Cria a pasta raiz
    os.makedirs(root_folder, exist_ok=True)
    
    # Executa as gerações
    create_directory_structure(root_folder)
    create_environment_config(root_folder)
    create_core_initializer(root_folder)
    create_trainer_engine(root_folder)
    create_stream_validator(root_folder)
    create_readme(root_folder)
    
    print("\\n[OK] Ambiente de desenvolvimento preparado com sucesso para a fase de treinamento/otimização!")
    
    # Exibe a árvore visual
    print_directory_tree(root_folder)

if __name__ == "__main__":
    main()
