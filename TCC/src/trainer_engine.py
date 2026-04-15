from ultralytics import YOLO

def load_yolo_model():
    print("Carregando o modelo YOLOv8 Nano...")
    # Inicializa o modelo YOLO usando os pesos nano padrão
    model = YOLO("yolov8n.pt")
    print("[OK] Modelo carregado com sucesso.")
    return model

if __name__ == "__main__":
    model = load_yolo_model()
