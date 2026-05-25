"""
server.py
Backend FastAPI com WebSockets para inferência em tempo real.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import json
import os

from collections import deque
from src.dialog.intent_mapper import IntentMapper

# Tentamos importar o modelo se existir
try:
    from tensorflow.keras.models import load_model
    # ─── Alteração 1: Atualizado para importar a nova camada customizada do professor ───
    from src.model.lstm_model import AttentionLayer
    import joblib
except ImportError:
    pass

app = FastAPI(title="LibrasAI API")

# Serve a pasta do frontend e static content
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Carregamento global de IA (se treinado)
MODEL_PATH = "models/libras_attention.h5"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

model = None
le = None
mapper = IntentMapper()

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    try:
        # ─── Alteração 2: Injetando a 'AttentionLayer' no mapeamento customizado do Keras ───
        model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})
        le = joblib.load(LABEL_ENCODER_PATH)
        print("✅ Modelo LSTM+AttentionLayer e LabelEncoder carregados com sucesso.")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo (continuará rodando sem predição): {e}")

# Janela deslizante para Smoothing (suavização temporal)
# Recomendação do usuário
predictions_window = deque(maxlen=5)

# Métricas
metrics = {
    "total_predictions": 0,
    "started": True
}

@app.get("/")
def get_dashboard():
    """Gera o arquivo index na raiz."""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.get("/metrics")
def get_metrics():
    """Retorna métricas em tempo real."""
    return metrics

@app.websocket("/ws/predict")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Conexão WebSocket iniciada...")
    
    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)
            
            sequence = np.array(payload["sequence"]) 
            
            # sequence formato esperado (30 frames, 42 features)
            if sequence.shape[0] != 30 or sequence.shape[1] != 42:
                # O formato está errado para a inferência!
                await ws.send_text(json.dumps({
                    "prediction": "INDEFINIDO",
                    "confidence": 0.0,
                    "response": "Erro de dimensão de features."
                }))
                continue

            metrics["total_predictions"] += 1

            if model and le:
                # Expandir dimensão do batch (1, 30, 42)
                seq_reshaped = np.expand_dims(sequence, axis=0)
                
                # Predição
                preds = model.predict(seq_reshaped, verbose=0)
                idx = int(np.argmax(preds))
                confidence = float(np.max(preds))
                
                if confidence >= 0.75:
                    label = str(le.inverse_transform([idx])[0])
                else:
                    label = "INDEFINIDO"

                # Smoothing temporal
                predictions_window.append(label)
                # Verifica a classe mais comum dos ultimos N frames
                final_label = max(set(predictions_window), key=predictions_window.count)
                
                final_conf = confidence if final_label == label else 0.0

            else:
                # Mock se o modelo não estiver criado
                final_label = "MOCK_SINAIS"
                final_conf = 0.99
            
            if final_label != "INDEFINIDO":
                response_data = mapper.get_response(final_label)
                text_response = response_data.get("resposta", "Entendido!")
            else:
                text_response = "Sinal fraco, tente novamente."

            res = {
                "prediction": final_label,
                "confidence": final_conf,
                "response": text_response
            }
            
            await ws.send_text(json.dumps(res))

    except WebSocketDisconnect:
        print("Conexão encerrada pelo cliente.")
    except Exception as e:
        print(f"Erro no Socket: {e}")
        try:
            await ws.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)