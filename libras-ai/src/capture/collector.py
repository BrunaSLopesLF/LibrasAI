"""
collector.py
Módulo de coleta interativa de amostras de sinais em Libras.

Uso:
    python -m src.capture.collector
"""

import os
import cv2
import numpy as np
import time

from src.detection.hand_detector import HandDetector

# Diretório base para salvar dados brutos
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Configurações da sequência
SEQUENCE_LENGTH = 30        # frames por sequência
NUM_SEQUENCES = 30          # sequências por sinal
COOLDOWN_SECONDS = 2        # pausa entre sequências


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def countdown(cap, frame_width, frame_height, seconds: int, message: str):
    """Exibe contagem regressiva na tela antes de iniciar a gravação."""
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, message, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, str(i), (frame_width // 2 - 40, frame_height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 200, 255), 8)
        cv2.imshow("LibrasAI — Coleta de Dados", frame)
        cv2.waitKey(1000)


def collect_sign(sign_name: str, num_sequences: int = NUM_SEQUENCES):
    """
    Coleta amostras de vídeo para um sinal específico.

    Args:
        sign_name: Nome do sinal (ex: 'OI', 'OBRIGADO').
        num_sequences: Número de sequências a coletar.
    """
    sign_dir = os.path.join(DATA_RAW_DIR, sign_name.upper())
    ensure_dir(sign_dir)

    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calcula o próximo índice de sequência disponível
    existing = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    start_seq = len(existing)

    print(f"\n[INFO] Coletando sinal: {sign_name.upper()}")
    print(f"[INFO] Sequências existentes: {start_seq}")
    print(f"[INFO] Novas sequências a gravar: {num_sequences}")
    print("[INFO] Pressione 'Q' para interromper a qualquer momento.\n")

    # Contagem regressiva inicial
    countdown(cap, frame_width, frame_height, 3, f"Prepare-se para sinalizar: {sign_name.upper()}")

    for seq_idx in range(start_seq, start_seq + num_sequences):
        sequence_frames = []

        # Pausa entre sequências
        if seq_idx > start_seq:
            countdown(cap, frame_width, frame_height, COOLDOWN_SECONDS,
                      f"Proxima sequencia em...")

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = detector.detect(frame)
            detector.draw_landmarks(frame, results)
            landmarks = detector.extract_landmarks(frame)
            sequence_frames.append(landmarks)

            # HUD (heads-up display)
            progress = int((frame_num + 1) / SEQUENCE_LENGTH * 100)
            bar_width = 300
            filled = int(bar_width * progress / 100)

            cv2.rectangle(frame, (30, frame_height - 60),
                          (30 + bar_width, frame_height - 35), (60, 60, 60), -1)
            cv2.rectangle(frame, (30, frame_height - 60),
                          (30 + filled, frame_height - 35), (0, 200, 100), -1)

            cv2.putText(frame, f"Sinal: {sign_name.upper()}", (30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(frame,
                        f"Seq {seq_idx - start_seq + 1}/{num_sequences}  "
                        f"Frame {frame_num + 1}/{SEQUENCE_LENGTH}",
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            if detector.has_hands(results):
                cv2.putText(frame, "MAO DETECTADA", (frame_width - 220, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

            cv2.imshow("LibrasAI — Coleta de Dados", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Coleta interrompida pelo usuário.")
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                return

        # Salva sequência
        if len(sequence_frames) == SEQUENCE_LENGTH:
            save_path = os.path.join(sign_dir, f"{seq_idx}.npy")
            np.save(save_path, np.array(sequence_frames))
            print(f"  ✔ Sequência {seq_idx - start_seq + 1}/{num_sequences} salva → {save_path}")
        else:
            print(f"  ✗ Sequência {seq_idx - start_seq + 1} incompleta, ignorada.")

    print(f"\n[CONCLUÍDO] {num_sequences} sequências coletadas para '{sign_name.upper()}'.")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


def main():
    print("=" * 50)
    print("   LibrasAI — Módulo de Coleta de Dados")
    print("=" * 50)

    signs_to_collect = [
        "OI", "OBRIGADO", "BOM_DIA", "SIM", "NAO",
        "AJUDA", "EU", "VOCE", "APRENDER", "LIBRAS"
    ]

    print("\nSinais disponíveis para coleta:")
    for i, s in enumerate(signs_to_collect, 1):
        sign_dir = os.path.join(DATA_RAW_DIR, s)
        count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")]) \
            if os.path.exists(sign_dir) else 0
        status = f"({count}/{NUM_SEQUENCES} seqs)" if count > 0 else "(vazio)"
        print(f"  {i:2}. {s:<12} {status}")

    print("\nOpções:")
    print("  Digite o número para coletar um sinal específico")
    print("  Digite 'todos' para coletar todos os sinais em sequência")
    print("  Digite 'sair' para encerrar")

    choice = input("\n> ").strip().lower()

    if choice == "sair":
        return
    elif choice == "todos":
        for sign in signs_to_collect:
            collect_sign(sign, NUM_SEQUENCES)
    elif choice.isdigit() and 1 <= int(choice) <= len(signs_to_collect):
        sign = signs_to_collect[int(choice) - 1]
        qtd = input(f"Quantas sequências para '{sign}'? [{NUM_SEQUENCES}]: ").strip()
        num = int(qtd) if qtd.isdigit() else NUM_SEQUENCES
        collect_sign(sign, num)
    else:
        # Tenta interpretar como nome de sinal
        sign_input = choice.upper()
        if sign_input in [s.upper() for s in signs_to_collect]:
            collect_sign(sign_input, NUM_SEQUENCES)
        else:
            print(f"[ERRO] Opção inválida: '{choice}'")


if __name__ == "__main__":
    main()
