import cv2
import torch
import numpy as np
from PIL import Image

def main():
    # Otimização do PyTorch para CPU
    torch.set_num_threads(4)
    print("Inicializando modelo ZoeDepth (Profundidade Métrica Absoluta)...")
    print("NOTA: O download inicial (~1.3 GB) pode demorar um pouco dependendo da sua internet.")
    
    # 1. Carregamento do ZoeDepth (Modelo Métrico)
    repo = "isl-org/ZoeDepth"
    # ZoeD_N é a versão padrão treinada em dados mistos (indoor/outdoor)
    zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoe = zoe.to(device)
    zoe.eval()
    
    print("Iniciando captura de vídeo. Pressione 'q' para sair.")
    cap = cv2.VideoCapture(2)
    # Resolução moderada para manter o FPS na CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # ALERTA BASEADO EM DISTÂNCIA REAL (Em metros)
    # Se a média da ROI for menor que 2 metros, soa o alarme.
    ALERT_DISTANCE_METERS = 2.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame da webcam.")
            break
            
        # O modelo ZoeDepth espera uma imagem em formato PIL (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 2. Inferência Métrica
        with torch.no_grad():
            # A mágica acontece aqui: infer_pil retorna um numpy array onde 
            # CADA PIXEL é literalmente a distância em METROS.
            depth_numpy = zoe.infer_pil(pil_img)
            
        # 3. Recorte da ROI (Terço Central)
        H, W = depth_numpy.shape
        y_start, y_end = 0, H
        x_start, x_end = int(W / 3), int(2 * W / 3)
        
        roi = depth_numpy[y_start:y_end, x_start:x_end]
        
        # 4. Heurística Métrica Absoluta
        # Agora a média não é uma cor aleatória (0-255), é a distância física em metros!
        roi_mean_dist = np.mean(roi)
        
        # Se a distância média do caminho for menor que nosso limite de segurança
        is_obstacle_near = roi_mean_dist < ALERT_DISTANCE_METERS
        
        # 5. Visualização (Transformando metros em Cores para podermos enxergar)
        # Vamos definir que 0 metros é a cor mais clara (255) e 8 metros é escuro (0).
        max_dist_to_visualize = 8.0 
        depth_vis = np.clip((1.0 - depth_numpy / max_dist_to_visualize) * 255.0, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        color_roi = (0, 0, 255) if is_obstacle_near else (0, 255, 0)
        cv2.rectangle(depth_colormap, (x_start, y_start), (x_end, y_end), color_roi, 3)
        
        display_frame = frame.copy()
        
        if is_obstacle_near:
            print(f"ALERTA: OBSTÁCULO A {roi_mean_dist:.2f} METROS!")
            cv2.putText(display_frame, f"OBSTACULO ({roi_mean_dist:.1f}m)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            # O caminho está livre. Mostra a que distância está o próximo objeto.
            cv2.putText(display_frame, f"LIVRE ({roi_mean_dist:.1f}m)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # Junta os dois frames para a janela dividida
        combined_view = np.hstack((display_frame, depth_colormap))
        
        # Trava o tamanho máximo para telas menores
        max_width = 1280
        if combined_view.shape[1] > max_width:
            scale = max_width / combined_view.shape[1]
            new_height = int(combined_view.shape[0] * scale)
            combined_view = cv2.resize(combined_view, (max_width, new_height))
            
        cv2.imshow('Percepcao Espacial - Metros Reais', combined_view)
        
        # Tecla 'q' para fechar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline métrico encerrado.")

if __name__ == "__main__":
    main()
