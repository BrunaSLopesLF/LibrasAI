"""
test_detector.py
Testes unitários para o módulo HandDetector.
"""

import os
import sys
import numpy as np
import pytest

# Garante que o root do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.hand_detector import HandDetector


@pytest.fixture(scope="module")
def detector():
    """Instância do HandDetector compartilhada entre testes."""
    d = HandDetector()
    yield d
    d.close()


def make_blank_frame(h=480, w=640) -> np.ndarray:
    """Cria frame preto para testes."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_skin_frame(h=480, w=640) -> np.ndarray:
    """Cria frame com cor de pele simulada (não detectará mão real)."""
    frame = np.full((h, w, 3), (150, 120, 100), dtype=np.uint8)
    return frame


class TestHandDetectorConstants:
    def test_feature_size(self):
        assert HandDetector.FEATURE_SIZE == 126

    def test_num_hands(self):
        assert HandDetector.NUM_HANDS == 2

    def test_num_landmarks(self):
        assert HandDetector.NUM_LANDMARKS == 21

    def test_num_coords(self):
        assert HandDetector.NUM_COORDS == 3


class TestExtractLandmarks:
    def test_output_shape_empty_frame(self, detector):
        """Frame sem mãos deve retornar vetor de zeros com shape correto."""
        frame = make_blank_frame()
        landmarks = detector.extract_landmarks(frame)
        assert landmarks.shape == (HandDetector.FEATURE_SIZE,)

    def test_output_dtype(self, detector):
        """Vetor de landmarks deve ser float32."""
        frame = make_blank_frame()
        landmarks = detector.extract_landmarks(frame)
        assert landmarks.dtype == np.float32

    def test_zero_padding_no_hands(self, detector):
        """Sem mãos detectadas, todo vetor deve ser zeros."""
        frame = make_blank_frame()
        landmarks = detector.extract_landmarks(frame)
        assert np.all(landmarks == 0.0)

    def test_values_in_range(self, detector):
        """Coordenadas normalizadas devem estar razoavelmente próximas de [0, 1]."""
        frame = make_blank_frame()
        landmarks = detector.extract_landmarks(frame)
        # Zeroes são OK; se houvesse mãos, checaríamos limites
        assert np.all(np.abs(landmarks) <= 5.0)


class TestDetect:
    def test_detect_returns_result(self, detector):
        """Método detect deve retornar objeto results sem erro."""
        frame = make_blank_frame()
        results = detector.detect(frame)
        assert results is not None

    def test_no_hands_in_blank_frame(self, detector):
        """Frame em branco não deve detectar mãos."""
        frame = make_blank_frame()
        results = detector.detect(frame)
        assert not detector.has_hands(results)


class TestDrawLandmarks:
    def test_draw_does_not_crash(self, detector):
        """Draw landmarks não deve levantar exceção."""
        frame = make_blank_frame()
        results = detector.detect(frame)
        result_frame = detector.draw_landmarks(frame.copy(), results)
        assert result_frame.shape == frame.shape

    def test_draw_returns_frame(self, detector):
        """Draw landmarks deve retornar um ndarray."""
        frame = make_blank_frame()
        results = detector.detect(frame)
        result_frame = detector.draw_landmarks(frame.copy(), results)
        assert isinstance(result_frame, np.ndarray)


class TestHasHands:
    def test_no_hands(self, detector):
        frame = make_blank_frame()
        results = detector.detect(frame)
        assert detector.has_hands(results) is False
