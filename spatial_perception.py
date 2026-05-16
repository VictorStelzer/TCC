import cv2
import torch
import numpy as np

def main():
    # Otimiza o PyTorch para rodar mais rápido na CPU
    torch.set_num_threads(4)
    print("Inicializando modelo MiDaS_small (pode levar alguns segundos na primeira vez)...")
    
    # 1. Carregamento do Modelo
    # Utiliza o MiDaS_small para garantir melhor taxa de quadros (FPS) e execução leve local.
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    # Move o modelo para o dispositivo adequado (GPU ou CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval() # Configura para modo de avaliação
    
    # Carrega as transformações esperadas pelo modelo MiDaS_small (redimensionamento e normalização)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    print("Iniciando captura de vídeo. Pressione 'q' para sair.")
    cap = cv2.VideoCapture(2)
    
    # Reduz a resolução nativa da webcam para 640x480 para eliminar o lag/travamento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Limiar de alerta: Reduzido para captar obstáculos mais distantes.
    # Valores menores disparam o alerta mais cedo (objetos mais longe).
    ALERT_THRESHOLD = 90
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler o frame da webcam.")
            break
            
        # O OpenCV lê em BGR. O modelo espera a imagem em RGB.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Prepara o frame
        # Aplica o transform do MiDaS e move para o device (CPU/GPU)
        input_batch = transform(img_rgb).to(device)
        
        # 3. Executa a inferência de profundidade
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Redimensiona o mapa de profundidade devolta para a resolução original do frame
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Move para a CPU e converte em matriz NumPy
        depth_map = prediction.cpu().numpy()
        
        # Normalização Robusta (Percentil 95):
        # Em vez de usar o valor máximo absoluto (que estraga a escala se um objeto 
        # muito próximo entrar nas bordas fora da ROI), usamos o percentil 95.
        # Isso estabiliza a escala térmica!
        d_min = np.min(depth_map)
        d_max = np.percentile(depth_map, 95)
        if d_max == d_min:
            d_max = d_min + 1e-5
            
        depth_map_normalized = np.clip((depth_map - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Recorte da Região de Interesse (ROI) - Terço central
        # A matemática da ROI é definida pelas proporções da imagem total:
        H, W = depth_map_normalized.shape
        # Altura: total da imagem
        y_start, y_end = 0, H
        # Largura: do índice correspondente a 1/3 (33%) até 2/3 (66%) da imagem (Caminho em frente)
        x_start, x_end = int(W / 3), int(2 * W / 3)
        
        # Isola os pixels da ROI utilizando fatiamento de arrays do numpy
        roi = depth_map_normalized[y_start:y_end, x_start:x_end]
        
        # 5. Heurística de Alerta
        # Calcula a média de intensidade dos pixels dentro da ROI
        roi_mean_depth = np.mean(roi)
        
        # Aplica um mapa de cores para o mapa de calor da profundidade (facilita a visualização)
        depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
        
        # Verifica se a média de proximidade excedeu o nosso limite seguro
        is_obstacle_near = roi_mean_depth > ALERT_THRESHOLD
        
        display_frame = frame.copy()
        
        if is_obstacle_near:
            # Emite alerta no terminal de texto
            print(f"ALERTA: OBSTÁCULO PRÓXIMO! (Intensidade: {roi_mean_depth:.1f})")
            
            # Desenha o texto de Alerta na tela do frame original e mapa de profundidade
            cv2.putText(display_frame, "OBSTACULO PROXIMO", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            color_roi = (0, 0, 255)  # Vermelho para alerta
        else:
            color_roi = (0, 255, 0)  # Verde se o caminho estiver livre
            
        # Desenha o retângulo que indica matematicamente a ROI no heatmap
        cv2.rectangle(depth_colormap, (x_start, y_start), (x_end, y_end), color_roi, 3)
        
        # 6. Exibição Dividida
        # Junta as imagens lado a lado
        combined_view = np.hstack((display_frame, depth_colormap))
        
        # Redimensiona a janela para evitar que fique maior que a tela do monitor
        max_width = 1280
        if combined_view.shape[1] > max_width:
            scale = max_width / combined_view.shape[1]
            new_height = int(combined_view.shape[0] * scale)
            combined_view = cv2.resize(combined_view, (max_width, new_height))
        
        # Abre a janela mostrando a visão dividida
        cv2.imshow('Percepcao Espacial - Frame Original vs Profundidade', combined_view)
        
        # Verifica se a tecla 'q' foi pressionada para fechar o loop graciosamente
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberação dos recursos de hardware
    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline de percepção espacial encerrado.")

if __name__ == "__main__":
    main()
