import cv2
import torch
import sys

def main():
    print("Verificando ambiente de percepção espacial...")
    print("-" * 40)
    
    # 1. Verificando PyTorch e CUDA
    print(f"Versão do PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"Suporte CUDA (Aceleração de Hardware) ativo: {cuda_available}")
    if cuda_available:
        print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: CUDA não disponível. O processamento será feito em CPU, o que pode aumentar a latência.")
    
    print("-" * 40)
    
    # 2. Verificando OpenCV e captura de vídeo
    print(f"Versão do OpenCV: {cv2.__version__}")
    print("Iniciando teste de captura da webcam...")
    print("Pressione 'q' na janela do vídeo para encerrar o teste.")
    
    # Inicia a captura da webcam padrão (índice 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a webcam.")
        sys.exit(1)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERRO: Falha ao capturar o frame da webcam.")
            break
            
        # Exibe o frame
        cv2.imshow('Teste de Percepcao - Pressione "q" para sair', frame)
        
        # Aguarda 1ms e verifica se 'q' foi pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Teste finalizado com sucesso.")

if __name__ == "__main__":
    main()
