"""
lstm_model.py
Definição da arquitetura LSTM com Attention para reconhecimento de sinais em Libras.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, LSTM, Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.regularizers import l2

# ─── Alteração 1: Nova Camada de Atenção Customizada do Professor ──────────────
class AttentionLayer(Layer):
    """
    Camada de Attention customizada fornecida pelo orientador.
    Calcula pesos de atenção locais e reduz a sequência temporal.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
            name="attention_weight"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Implementação matemática baseada em multiplicação de matrizes (matmul)
        score = tf.matmul(inputs, self.W)
        weights = tf.nn.softmax(score, axis=1)

        context = weights * inputs
        context = tf.reduce_sum(context, axis=1)
        return context
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def build_lstm_model(num_classes: int,
                     sequence_length: int = 30,
                     feature_size: int = 42,
                     dropout_rate: float = 0.3) -> Model:
    """
    Constrói e retorna o modelo LSTM+Attention corrigido.

    Arquitetura Atualizada:
        Input(30, 42)
        → LSTM(128, return_sequences=True) + Dropout
        → LSTM(64, return_sequences=True) + BatchNorm
        → AttentionLayer()  [Customizada pelo Professor]
        → Dense(64, relu) + Dropout
        → Dense(num_classes, softmax)
    """
    inputs = Input(shape=(sequence_length, feature_size), name="input_frames")

    x = LSTM(128, return_sequences=True, 
             kernel_regularizer=l2(1e-4),
             recurrent_regularizer=l2(1e-4))(inputs)
    x = Dropout(dropout_rate)(x)

    x = LSTM(64, return_sequences=True,
             kernel_regularizer=l2(1e-4),
             recurrent_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    # ─── Alteração 2: Aplicação da Nova Atenção Customizada ───────────────────
    x = AttentionLayer()(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation="softmax", name="output_classes")(x)

    model = Model(inputs, outputs, name="LibrasAI_LSTM_Attention")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def print_model_summary(num_classes: int = 10):
    """Exibe o sumário da arquitetura do modelo."""
    model = build_lstm_model(num_classes)
    model.summary()
    return model

if __name__ == "__main__":
    print_model_summary()