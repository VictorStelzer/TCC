import os
import sys
import subprocess
import time
import threading
from queue import Queue
from datetime import datetime

# ======= 1. GERENCIAMENTO DE DEPENDÊNCIAS =======
def check_dependencies():
    """Identifica e instala pacotes ausentes via pip."""
    packages = {
        "ultralytics": "ultralytics",
        "cv2": "opencv-python",
        "cap_from_youtube": "cap-from-youtube"
    }

    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"[TCC] Módulo '{package_name}' não encontrado. Instalando...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

check_dependencies()

# ======= 2. IMPORTAÇÕES PÓS-VERIFICAÇÃO =======
import cv2
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO

# ======= 3. FUNÇÃO DE REGISTRO E LOG =======
def save_log(success, message):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    log_path = os.path.join(outputs_dir, "stream_test_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCESSO" if success else "ERRO"
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{status}] {message}\n")

# ======= 4. NÚCLEO DO VALIDADOR DE STREAM (MULTITHREADING) =======
def run_realtime_inference():
    print("[TCC] Carregando o modelo YOLOv8 Nano...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        msg = f"Falha ao carregar modelo YOLO: {e}"
        print(f"[ERRO] {msg}")
        save_log(False, msg)
        return

    print("\n" + "="*40)
    print(" 🎥 SELECIONE A FONTE DE VÍDEO")
    print("="*40)
    print(" [1] Vídeo do YouTube")
    print(" [2] Webcam Local")
    print("="*40)
    
    escolha = input("Digite 1 ou 2 para escolher a fonte: ").strip()
    fonte_usada = ""

    if escolha == '2':
        print("\n[TCC] 📷 Abrindo Câmera Local...")
        cap = cv2.VideoCapture(0)
        fonte_usada = "Webcam Local"
        if not cap.isOpened():
            print("[ERRO] Não foi possível acessar a webcam.")
            return
    else:
        # Usando link configurado pelo usuário no último edit
        youtube_url = "https://www.youtube.com/watch?v=6y5CqAHxGX0"
        print(f"\n[TCC] 🌐 Conectando na stream do YouTube ({youtube_url})...")
        fonte_usada = f"YouTube: {youtube_url}"
        try:
            cap = cap_from_youtube(youtube_url, resolution='best')
            if cap is None or not cap.isOpened():
                raise ValueError("O cv2.VideoCapture retornou nulo ou inativo.")
        except Exception as e:
            msg = f"Não foi possível processar a URL. Detalhe: {e}"
            print(f"[ERRO] {msg}")
            save_log(False, msg)
            return

    msg_sucesso = "Conexão com a fonte estabelecida e validada!"
    print(f"[TCC] {msg_sucesso}")
    print("[TCC] Iniciando Thread Assíncrona de IA para fluidez do vídeo...")

    # ======= ARQUITETURA MULTITHREADING (IA EM SEGUNDO PLANO) =======
    # Isolamos o OpenCV num thread e o YOLO em outro! 
    # Dessa forma o CPU não freia a câmera enquanto pensa, extinguindo as "travadas" e o jitter de FPS descritos.
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    
    def ai_worker():
        while not stop_event.is_set():
            if not frame_queue.empty():
                input_frame = frame_queue.get()
                # Processa a rede no menor peso viável para processador (resolução interna de 320px)
                results = model.predict(input_frame, imgsz=320, conf=0.4, verbose=False)
                
                # Atualiza os resultados apagando os defasados (Esvazia a fila)
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except:
                        pass
                
                # Injeta a inferência na visualização
                result_queue.put(results[0])
            else:
                time.sleep(0.005)

    thread_ia = threading.Thread(target=ai_worker, daemon=True)
    thread_ia.start()

    target_width = 640
    curr_results = None
    
    # Para estabilizar matematicamente o texto na tela
    fps_history = []
    prev_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n[TCC] Transmissão interrompida ou fim do fluxo alcançado.")
                break
                
            orig_h, orig_w = frame.shape[:2]
            ratio = target_width / orig_w
            target_height = int(orig_h * ratio)
            
            frame_resized = cv2.resize(frame, (target_width, target_height))
            annotated_frame = frame_resized.copy()
            
            # Se a Thread IA terminou de pensar, nós mandamos a próxima imagem
            if frame_queue.empty():
                frame_queue.put(frame_resized.copy())
                
            # Puxamos as caixas mais recentes descobertas na Async Thread
            if not result_queue.empty():
                curr_results = result_queue.get()
                
            # Desenha assincronamente as caixas no vídeo fluido
            if curr_results is not None and len(curr_results.boxes) > 0:
                for box in curr_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # ====== CÁLCULO DE FPS ESTÁVEL (Média Móvel) ======
            current_time = time.time()
            fps_instant = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Evita o glitch de oscilação 60 -> 3 -> 60 -> ... aplicando uma média dos últimos 25 quadros
            fps_history.append(fps_instant)
            if len(fps_history) > 25: 
                fps_history.pop(0)
                
            fps_avg = sum(fps_history) / len(fps_history)
            
            # Exibe o FPS garantido e polido
            fps_text = f"FPS (Fluido): {fps_avg:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.imshow("TCC: Simulacao POV Edge (Multithread)", annotated_frame)
            
            # Retorno real-time constante e espera do input (q para sair)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[TCC] Visualização desativada ('q' pressionado).")
                break
                
        save_log(True, f"Teste finalizado de maneira saudável. Fonte processada: {fonte_usada}")
        
    except Exception as e:
        msg = f"Erro inesperado durante processamento em tempo real dos frames: {e}"
        print(f"\n[ERRO] {msg}")
        save_log(False, msg)
        
    finally:
        stop_event.set()
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[TCC] Monitoramento Finalizado. Veja os logs em /outputs.")

if __name__ == "__main__":
    run_realtime_inference()
