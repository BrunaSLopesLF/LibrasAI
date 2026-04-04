"""
test_predictor.py
Testes unitários para o módulo Predictor (sem modelo real).
Usa um modelo mockado para validar o comportamento do buffer e threshold.
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126
CLASS_NAMES = ["OI", "OBRIGADO", "SIM", "NAO", "LIBRAS"]
NUM_CLASSES = len(CLASS_NAMES)


def make_fake_model(predicted_class: int = 0, confidence: float = 0.9):
    """Cria um modelo keras mockado que retorna probabilidades fixas."""
    model = MagicMock()
    probs = np.zeros((1, NUM_CLASSES))
    probs[0, predicted_class] = confidence
    probs[0, (predicted_class + 1) % NUM_CLASSES] = 1 - confidence
    model.predict.return_value = probs
    model.output_shape = (None, NUM_CLASSES)
    return model


def make_metadata(tmp_dir: str) -> str:
    """Cria arquivo de metadados temporário."""
    metadata = {
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": FEATURE_SIZE,
        "confidence_threshold": 0.70,
    }
    path = os.path.join(tmp_dir, "model_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    return path


@pytest.fixture
def predictor_with_mock():
    """Predictor com modelo mockado e metadados temporários."""
    from src.model.predictor import Predictor

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_path = make_metadata(tmp_dir)
        model_path = os.path.join(tmp_dir, "libras_lstm.h5")

        # Cria arquivo .h5 vazio para simular existência
        open(model_path, "w").close()

        predictor = Predictor.__new__(Predictor)
        predictor.threshold = 0.70
        predictor.class_names = CLASS_NAMES
        predictor.metadata = {}
        from collections import deque
        predictor.buffer = deque(maxlen=SEQUENCE_LENGTH)
        predictor.model = make_fake_model(predicted_class=0, confidence=0.95)

        yield predictor


class TestBuffer:
    def test_buffer_empty_initially(self, predictor_with_mock):
        p = predictor_with_mock
        assert not p.is_ready()

    def test_buffer_progress_zero(self, predictor_with_mock):
        p = predictor_with_mock
        assert p.buffer_progress == 0.0

    def test_buffer_fills_up(self, predictor_with_mock):
        p = predictor_with_mock
        for _ in range(SEQUENCE_LENGTH):
            p.add_frame(np.zeros(FEATURE_SIZE, dtype=np.float32))
        assert p.is_ready()

    def test_buffer_progress_full(self, predictor_with_mock):
        p = predictor_with_mock
        for _ in range(SEQUENCE_LENGTH):
            p.add_frame(np.zeros(FEATURE_SIZE, dtype=np.float32))
        assert p.buffer_progress == 1.0

    def test_reset_clears_buffer(self, predictor_with_mock):
        p = predictor_with_mock
        for _ in range(10):
            p.add_frame(np.zeros(FEATURE_SIZE, dtype=np.float32))
        p.reset_buffer()
        assert not p.is_ready()
        assert p.buffer_progress == 0.0


class TestPredict:
    def _fill_buffer(self, p):
        for _ in range(SEQUENCE_LENGTH):
            p.add_frame(np.random.rand(FEATURE_SIZE).astype(np.float32))

    def test_predict_returns_none_when_not_ready(self, predictor_with_mock):
        p = predictor_with_mock
        result = p.predict()
        assert result is None

    def test_predict_returns_sign_above_threshold(self, predictor_with_mock):
        p = predictor_with_mock
        p.model = make_fake_model(predicted_class=0, confidence=0.95)
        self._fill_buffer(p)
        result = p.predict()
        assert result is not None
        sign, confidence = result
        assert sign == CLASS_NAMES[0]
        assert confidence == pytest.approx(0.95, abs=0.01)

    def test_predict_returns_none_below_threshold(self, predictor_with_mock):
        p = predictor_with_mock
        p.model = make_fake_model(predicted_class=0, confidence=0.50)
        p.threshold = 0.70
        self._fill_buffer(p)
        result = p.predict()
        assert result is None

    def test_predict_returns_correct_class(self, predictor_with_mock):
        p = predictor_with_mock
        p.model = make_fake_model(predicted_class=2, confidence=0.88)  # SIM
        self._fill_buffer(p)
        result = p.predict()
        assert result is not None
        assert result[0] == "SIM"


class TestTopPredictions:
    def test_top_predictions_empty_buffer(self, predictor_with_mock):
        p = predictor_with_mock
        assert p.get_top_predictions() == []

    def test_top_predictions_returns_n_items(self, predictor_with_mock):
        p = predictor_with_mock
        for _ in range(SEQUENCE_LENGTH):
            p.add_frame(np.random.rand(FEATURE_SIZE).astype(np.float32))
        tops = p.get_top_predictions(n=3)
        assert len(tops) == 3

    def test_top_predictions_sorted_by_confidence(self, predictor_with_mock):
        p = predictor_with_mock
        for _ in range(SEQUENCE_LENGTH):
            p.add_frame(np.random.rand(FEATURE_SIZE).astype(np.float32))
        tops = p.get_top_predictions(n=3)
        confidences = [c for _, c in tops]
        assert confidences == sorted(confidences, reverse=True)
