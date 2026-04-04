"""
helpers.py
Utilitários gerais do LibrasAI.
"""

import os
import json
import time
import numpy as np
from typing import Optional


def fps_counter():
    """
    Gerador para calcular FPS em tempo real.

    Uso:
        counter = fps_counter()
        fps = next(counter)
    """
    prev_time = time.time()
    while True:
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        yield round(fps, 1)


def load_model_metadata(models_dir: str) -> Optional[dict]:
    """Carrega metadados do modelo treinado."""
    path = os.path.join(models_dir, "model_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_exists(models_dir: str) -> bool:
    """Verifica se o modelo treinado existe."""
    return os.path.exists(os.path.join(models_dir, "libras_lstm.h5"))


def count_training_samples(data_raw_dir: str) -> dict:
    """
    Conta amostras disponíveis por sinal.

    Returns:
        Dicionário {sinal: quantidade}.
    """
    counts = {}
    if not os.path.exists(data_raw_dir):
        return counts

    for sign in os.listdir(data_raw_dir):
        sign_dir = os.path.join(data_raw_dir, sign)
        if os.path.isdir(sign_dir):
            npy_files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
            counts[sign] = len(npy_files)

    return counts


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normaliza landmarks em relação ao pulso (ponto 0).
    Opcional — para robustez adicional ao deslocamento.

    Args:
        landmarks: Array (126,) de coordenadas.

    Returns:
        Array (126,) normalizado.
    """
    if landmarks.sum() == 0:
        return landmarks

    normalized = landmarks.copy()
    # Normaliza cada mão separadamente
    for hand_idx in range(2):
        offset = hand_idx * 63  # 21 pontos × 3 coords
        wrist = normalized[offset:offset + 3].copy()
        if wrist.sum() != 0:  # Se a mão está presente
            for pt_idx in range(21):
                base = offset + pt_idx * 3
                normalized[base] -= wrist[0]      # x
                normalized[base + 1] -= wrist[1]  # y
                normalized[base + 2] -= wrist[2]  # z

    return normalized


def format_confidence(confidence: float) -> str:
    """Formata confiança como percentual legível."""
    return f"{confidence * 100:.1f}%"


def get_sign_emoji(sign: str) -> str:
    """Retorna emoji associado ao sinal."""
    emojis = {
        "OI": "👋", "OBRIGADO": "🙏", "BOM_DIA": "☀️",
        "SIM": "✅", "NAO": "❌", "AJUDA": "🤝",
        "EU": "👤", "VOCE": "👉", "APRENDER": "📚", "LIBRAS": "🤟",
    }
    return emojis.get(sign.upper(), "🤙")
