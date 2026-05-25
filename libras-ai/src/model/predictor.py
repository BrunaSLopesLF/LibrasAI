"""
predictor.py
Inferência em tempo real com buffer circular de sequências.
"""

import os
import json
import numpy as np
from collections import deque
from typing import Optional, Tuple

import tensorflow as tf

# ─── Importar a nova camada customizada do professor ─────────────────────────
from src.model.lstm_model import AttentionLayer

# Suprimir logs do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "libras_lstm.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

SEQUENCE_LENGTH = 30
DEFAULT_THRESHOLD = 0.70


class Predictor:
    """
    Realiza inferência em tempo real usando o modelo LSTM treinado.

    Mantém um buffer circular de 30 frames. Quando o buffer está cheio,
    executa a predição e retorna o sinal com maior probabilidade, desde
    que a confiança seja maior que o threshold definido.
    """

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 metadata_path: str = METADATA_PATH,
                 threshold: float = DEFAULT_THRESHOLD):
        """
        Args:
            model_path: Caminho para o arquivo .h5 do modelo.
            metadata_path: Caminho para o JSON de metadados.
            threshold: Confiança mínima para aceitar predição (0.0 a 1.0).
        """
        self.threshold = threshold
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.model = None
        self.class_names = []
        self.metadata = {}

        self._load_model(model_path, metadata_path)

    def _load_model(self, model_path: str, metadata_path: str):
        """Carrega modelo e injeta mapeamento de classes fixo."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                "Execute o treinamento primeiro: python -m src.model.trainer"
            )

        # Injetando a AttentionLayer nos custom_objects do Keras
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'AttentionLayer': AttentionLayer}
        )

        # ─── CORREÇÃO CIRÚRGICA: Mapeamento Estático dos Sinais Oficiais ────────
        # Forçamos a lista ordenada para garantir a sincronia com o LabelEncoder
        self.class_names = [
            "AJUDA", "APRENDER", "BOM_DIA", "EU", "LIBRAS", 
            "NAO", "OBRIGADO", "OI", "SIM", "VOCE"
        ]
        
        # Tenta ler o threshold do JSON se ele existir, caso contrário usa o padrão
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.threshold = self.metadata.get("confidence_threshold", DEFAULT_THRESHOLD)
            except Exception:
                self.threshold = DEFAULT_THRESHOLD
        else:
            self.threshold = DEFAULT_THRESHOLD

        print(f"[Predictor] Modelo carregado com sucesso.")
        print(f"[Predictor] Mapeamento de classes fixado: {self.class_names}")
        print(f"[Predictor] Threshold ativo: {self.threshold:.0%}")

    def add_frame(self, landmarks: np.ndarray):
        """
        Adiciona um frame ao buffer.

        Args:
            landmarks: Vetor de features do frame (shape: (126,)).
        """
        self.buffer.append(landmarks)

    def is_ready(self) -> bool:
        """Retorna True quando o buffer está cheio (30 frames)."""
        return len(self.buffer) == SEQUENCE_LENGTH

    def predict(self) -> Optional[Tuple[str, float]]:
        """
        Executa predição com os frames no buffer.

        Returns:
            Tupla (nome_do_sinal, confiança) se confiança >= threshold.
            None se abaixo do threshold ou buffer incompleto.
        """
        if not self.is_ready():
            return None

        # Convertemos o buffer temporal para um array estruturado
        buffer_array = np.array(self.buffer)  # Shape original: (30, 126)

        # Slicing de dimensões (Foco nas 42 features da mão principal)
        if buffer_array.shape[1] == 126:
            buffer_array = buffer_array[:, :42]  # Reduz para shape: (30, 42)

        sequence = np.expand_dims(buffer_array, axis=0)  # Shape final: (1, 30, 42)
        probabilities = self.model.predict(sequence, verbose=0)[0]

        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])

        if confidence >= self.threshold:
            # Busca o nome real na nossa lista tratada
            sign_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"CLASSE_{class_idx}"
            return sign_name, confidence

        return None

    def get_top_predictions(self, n: int = 3) -> list:
        """
        Retorna as N predições com maior probabilidade, independente do threshold.

        Args:
            n: Número de predições a retornar.

        Returns:
            Lista de tuplas (sinal, confiança) ordenada por confiança.
        """
        if not self.is_ready():
            return []

        buffer_array = np.array(self.buffer)

        # Slicing preventivo também no fluxo de métricas secundárias
        if buffer_array.shape[1] == 126:
            buffer_array = buffer_array[:, :42]

        sequence = np.expand_dims(buffer_array, axis=0)
        probabilities = self.model.predict(sequence, verbose=0)[0]

        top_n = np.argsort(probabilities)[::-1][:n]
        return [
            (self.class_names[i] if i < len(self.class_names) else f"CLASSE_{i}",
             float(probabilities[i]))
            for i in top_n
        ]

    def reset_buffer(self):
        """Limpa o buffer de frames."""
        self.buffer.clear()

    @property
    def buffer_progress(self) -> float:
        """Progresso do preenchimento do buffer (0.0 a 1.0)."""
        return len(self.buffer) / SEQUENCE_LENGTH