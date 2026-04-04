"""
hand_detector.py
Wrapper de detecção de mãos compatível com MediaPipe (API legada via solutions).
Suporta MediaPipe >= 0.10.9 com mp.solutions.hands disponível.
"""

import cv2
import numpy as np

# Importação com fallback para compatibilidade
try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    _mp_drawing = mp.solutions.drawing_utils
    # drawing_styles pode não estar disponível em todas as versões
    try:
        _mp_drawing_styles = mp.solutions.drawing_styles
        _HAS_DRAWING_STYLES = True
    except AttributeError:
        _mp_drawing_styles = None
        _HAS_DRAWING_STYLES = False
    _MEDIAPIPE_AVAILABLE = True
except (AttributeError, ImportError):
    _MEDIAPIPE_AVAILABLE = False


class HandDetector:
    """
    Detecta mãos em frames de vídeo e extrai coordenadas normalizadas.

    Retorna um vetor de 126 features:
      - 21 pontos × 3 coordenadas (x, y, z) × 2 mãos
      - Zero-padding quando uma ou ambas as mãos estão ausentes
    """

    NUM_HANDS = 2
    NUM_LANDMARKS = 21
    NUM_COORDS = 3  # x, y, z
    FEATURE_SIZE = NUM_HANDS * NUM_LANDMARKS * NUM_COORDS  # 126

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe (mp.solutions) não disponível. "
                "Instale: pip install mediapipe==0.10.9"
            )

        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.NUM_HANDS,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray):
        """
        Executa detecção de mãos em um frame BGR.

        Args:
            frame: Frame BGR do OpenCV (np.ndarray).

        Returns:
            Objeto results do MediaPipe.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        return results

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Extrai vetor de landmarks normalizado do frame.

        Args:
            frame: Frame BGR do OpenCV.

        Returns:
            np.ndarray de shape (126,) com coordenadas normalizadas.
            Preenche com zeros quando mão não detectada.
        """
        results = self.detect(frame)
        landmarks = np.zeros(self.FEATURE_SIZE, dtype=np.float32)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= self.NUM_HANDS:
                    break
                offset = hand_idx * self.NUM_LANDMARKS * self.NUM_COORDS
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    base = offset + lm_idx * self.NUM_COORDS
                    landmarks[base]     = lm.x
                    landmarks[base + 1] = lm.y
                    landmarks[base + 2] = lm.z

        return landmarks

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Desenha landmarks e conexões sobre o frame.

        Args:
            frame: Frame BGR do OpenCV.
            results: Resultado da detecção MediaPipe.

        Returns:
            Frame com landmarks desenhados.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if _HAS_DRAWING_STYLES:
                    _mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        _mp_hands.HAND_CONNECTIONS,
                        _mp_drawing_styles.get_default_hand_landmarks_style(),
                        _mp_drawing_styles.get_default_hand_connections_style(),
                    )
                else:
                    # Fallback sem estilos
                    _mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        _mp_hands.HAND_CONNECTIONS,
                    )
        return frame

    def has_hands(self, results) -> bool:
        """Verifica se alguma mão foi detectada."""
        return results.multi_hand_landmarks is not None

    def close(self):
        """Libera recursos do MediaPipe."""
        self.hands.close()
