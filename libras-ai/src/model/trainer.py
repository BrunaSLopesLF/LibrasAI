"""
trainer.py
Pipeline completo de treinamento do modelo LSTM+Attention para LibrasAI.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import seaborn as sns

from src.model.lstm_model import build_lstm_model

# Caminhos
BASE_DIR = os.path.dirname(__file__)
DATA_RAW_DIR = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")

# Hiperparâmetros
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 42
EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
CONFIDENCE_THRESHOLD = 0.70

def load_data(data_dir: str):
    """Carrega as sequências e lida o LabelEncoder via joblib."""
    X, y_raw = [], []
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not class_names:
        raise ValueError(f"[ERRO] Nenhum sinal encontrado em {data_dir}.")

    for sign in class_names:
        sign_dir = os.path.join(data_dir, sign)
        sequences = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]

        for seq_file in sequences:
            path = os.path.join(sign_dir, seq_file)
            seq = np.load(path)
            
            # Pega as primeiras 42 posições (x, y de 21 landmarks) caso as antigas sejam 126
            if seq.shape[-1] >= 42:
                seq = seq[:, :42]
            
            if seq.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                X.append(seq)
                y_raw.append(sign)

    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    # Salvar LabelEncoder
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    return X, y_encoded, le, le.classes_.tolist()

def plot_training_history(history, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Histórico de Treinamento - LSTM+Attention", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set_title("Loss")
    axes[1].legend()

    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names: list, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    X, y, le, class_names = load_data(DATA_RAW_DIR)
    num_classes = len(class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    model = build_lstm_model(num_classes, feature_size=FEATURE_SIZE)
    model.summary()

    model_path = os.path.join(MODELS_DIR, "libras_attention.h5")
    
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Meta atingida? {'Sim' if accuracy >= 0.8 else 'Nao'} - Acc: {accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(classification_report(y_val, y_pred, target_names=class_names))

    plot_training_history(history, MODELS_DIR)
    plot_confusion_matrix(y_val, y_pred, class_names, MODELS_DIR)

if __name__ == "__main__":
    train()
