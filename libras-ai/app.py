"""
app.py
LibrasAI — Interface principal com Streamlit.

Uso:
    streamlit run app.py
"""

import os
import sys
import cv2
import numpy as np
import streamlit as st
import time
from datetime import datetime

# ─── Configuração de página (deve ser o primeiro comando Streamlit) ──────────
st.set_page_config(
    page_title="LibrasAI — Assistente de Libras",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

# ─── CSS Customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Importar fonte ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Reset e variáveis ────────────────────────────── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2035;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: rgba(255,255,255,0.08);
    --glass: rgba(255,255,255,0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ─── Header ──────────────────────────────────────── */
.libras-header {
    background: linear-gradient(135deg, #1a1040 0%, #0d1f3c 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.libras-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.libras-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.libras-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
}

/* ─── Cards de métricas ───────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(139,92,246,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139,92,246,0.15);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-value.accent-blue  { color: var(--accent-blue);   }
.metric-value.accent-green { color: var(--accent-green);  }
.metric-value.accent-purple{ color: var(--accent-purple); }
.metric-value.accent-cyan  { color: var(--accent-cyan);   }

/* ─── Painel de sinal detectado ───────────────────── */
.sign-panel {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.12));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.sign-name {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em;
}
.sign-confidence {
    font-size: 0.9rem;
    color: var(--accent-green);
    font-weight: 600;
    margin-top: 4px;
}
.sign-waiting {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* ─── Histórico de conversa ──────────────────────── */
.chat-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    max-height: 380px;
    overflow-y: auto;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 2px; }

.chat-message {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
.chat-sinal {
    font-size: 0.75rem;
    color: var(--accent-purple);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.chat-text {
    font-size: 0.9rem;
    color: var(--text-primary);
    line-height: 1.5;
}
.chat-time {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 6px;
}

/* ─── Barra de progresso do buffer ──────────────── */
.buffer-bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.buffer-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    transition: width 0.1s ease;
}

/* ─── Status badges ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-active  { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-stopped { background: rgba(239,68,68,0.2);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3);  }
.badge-warning { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

/* ─── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ─── Botões ─────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* ─── Dica do sinal ────────────────────────────── */
.dica-box {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--accent-cyan);
    margin-top: 8px;
}

/* ─── Seção de sinais suportados ──────────────── */
.sign-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-top: 8px;
}
.sign-chip {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-align: center;
}

/* ─── Streamlit overrides ─────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important; }
div[data-testid="stVideo"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Verificar disponibilidade do modelo ──────────────────────────────────────
def check_model_available() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "libras_lstm.h5"))


def load_predictor_cached():
    """Carrega o predictor com cache do Streamlit."""
    from src.model.predictor import Predictor
    return Predictor(threshold=st.session_state.get("threshold", 0.70))


# ─── Estado da sessão ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "camera_active": False,
        "conversation_log": [],
        "last_sign": None,
        "last_confidence": 0.0,
        "last_response": None,
        "threshold": 0.70,
        "buffer_progress": 0.0,
        "fps": 0.0,
        "total_detections": 0,
        "session_start": datetime.now().strftime("%H:%M:%S"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤟 LibrasAI")
    st.markdown("Assistente inteligente de conversação em Libras")
    st.divider()

    st.markdown("#### ⚙️ Configurações")
    threshold = st.slider(
        "Limiar de confiança",
        min_value=0.50, max_value=0.95, value=0.70, step=0.05,
        help="Confiança mínima para aceitar um sinal"
    )
    st.session_state["threshold"] = threshold
    st.caption(f"Predições abaixo de {threshold:.0%} serão ignoradas")

    st.divider()

    # Status do modelo
    st.markdown("#### 📊 Status do Sistema")
    if check_model_available():
        st.markdown('<span class="badge badge-active">✓ Modelo Carregado</span>', unsafe_allow_html=True)

        import json
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.caption(f"Classes: {meta.get('num_classes', '?')}")
            st.caption(f"Acurácia: {meta.get('best_val_accuracy', 0)*100:.1f}%")
            st.caption(f"Treinado em: {meta.get('timestamp', 'N/A')[:10]}")
    else:
        st.markdown('<span class="badge badge-warning">⚠ Modelo não encontrado</span>', unsafe_allow_html=True)
        st.caption("Execute: `python -m src.model.trainer`")

    st.divider()

    # Sinais suportados
    st.markdown("#### 🤙 Sinais Suportados")
    signs = ["OI 👋", "OBRIGADO 🙏", "BOM DIA ☀️", "SIM ✅",
             "NÃO ❌", "AJUDA 🤝", "EU 👤", "VOCÊ 👉",
             "APRENDER 📚", "LIBRAS 🤟"]
    sign_html = '<div class="sign-grid">'
    for s in signs:
        sign_html += f'<div class="sign-chip">{s}</div>'
    sign_html += '</div>'
    st.markdown(sign_html, unsafe_allow_html=True)

    st.divider()

    # Estatísticas da sessão
    st.markdown("#### 📈 Sessão Atual")
    st.caption(f"Iniciada às: {st.session_state['session_start']}")
    st.caption(f"Detecções bem-sucedidas: {st.session_state['total_detections']}")

    if st.button("🔄 Reiniciar Sessão", use_container_width=True):
        st.session_state["conversation_log"] = []
        st.session_state["total_detections"] = 0
        st.session_state["last_sign"] = None
        st.session_state["last_response"] = None
        st.session_state["session_start"] = datetime.now().strftime("%H:%M:%S")
        st.rerun()


# ─── Header principal ─────────────────────────────────────────────────────────
st.markdown("""
<div class="libras-header">
    <h1>🤟 LibrasAI</h1>
    <p>Assistente Inteligente de Conversação em Libras — Aprenda com IA em tempo real</p>
</div>
""", unsafe_allow_html=True)


# ─── Verificação do modelo ─────────────────────────────────────────────────────
if not check_model_available():
    st.warning("""
    **⚠️ Modelo não treinado ainda!**

    Para usar o LibrasAI, siga estes passos:

    1. **Coletar dados** → `python -m src.capture.collector`
    2. **Treinar modelo** → `python -m src.model.trainer`
    3. **Voltar aqui** → Reload da página
    """)

    st.info("""
    💡 **Modo Demonstração:** Mesmo sem o modelo treinado, você pode explorar a interface.
    O sistema irá mostrar a detecção de mãos via MediaPipe em tempo real.
    """)

# ─── Layout principal ─────────────────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("#### 📷 Câmera ao Vivo")

    # Controles da câmera
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        if not st.session_state["camera_active"]:
            if st.button("▶ Iniciar Câmera", use_container_width=True, type="primary"):
                st.session_state["camera_active"] = True
                st.rerun()
        else:
            if st.button("⏹ Parar Câmera", use_container_width=True):
                st.session_state["camera_active"] = False
                st.rerun()
    with ctrl_col2:
        status_txt = "🟢 Câmera Ativa" if st.session_state["camera_active"] else "🔴 Câmera Parada"
        st.markdown(f"<div style='padding:8px 0; color: {'#10b981' if st.session_state['camera_active'] else '#ef4444'}; font-weight:600;'>{status_txt}</div>",
                    unsafe_allow_html=True)

    # Placeholder do feed de vídeo
    video_placeholder = st.empty()
    buffer_placeholder = st.empty()

    # Métricas em linha
    m1, m2, m3 = st.columns(3)
    with m1:
        fps_placeholder = st.empty()
    with m2:
        conf_placeholder = st.empty()
    with m3:
        buf_pct_placeholder = st.empty()


with col_info:
    st.markdown("#### 🎯 Sinal Detectado")
    sign_placeholder = st.empty()

    st.markdown("#### 💬 Conversa")
    chat_placeholder = st.empty()

    # Dica do último sinal
    st.markdown("#### 💡 Feedback")
    dica_placeholder = st.empty()


# ─── Loop principal de vídeo ──────────────────────────────────────────────────
def render_sign_panel(sign: str = None, confidence: float = 0.0):
    if sign:
        from src.utils.helpers import get_sign_emoji
        emoji = get_sign_emoji(sign)
        return f"""
        <div class="sign-panel">
            <div style="font-size:2.5rem;">{emoji}</div>
            <div class="sign-name">{sign.replace('_', ' ')}</div>
            <div class="sign-confidence">Confiança: {confidence*100:.1f}%</div>
        </div>"""
    else:
        return """
        <div class="sign-panel">
            <div class="sign-waiting">🤙 Aguardando sinal...</div>
            <div style="font-size:0.8rem;color:#475569;margin-top:8px;">
                Posicione as mãos na frente da câmera
            </div>
        </div>"""


def render_chat(log: list):
    if not log:
        return """
        <div class="chat-container">
            <div style="text-align:center;color:#475569;padding:24px;font-size:0.85rem;">
                O histórico de conversa aparecerá aqui...<br>
                Comece a sinalizar! 🤟
            </div>
        </div>"""

    messages_html = ""
    for entry in log[-8:]:
        messages_html += f"""
        <div class="chat-message">
            <div class="chat-sinal">{entry['emoji']} {entry['sinal']}</div>
            <div class="chat-text">{entry['resposta']}</div>
            <div class="chat-time">{entry['timestamp']}</div>
        </div>"""

    return f'<div class="chat-container">{messages_html}</div>'


def render_dica(dica: str = None):
    if dica:
        return f'<div class="dica-box">ℹ️ {dica}</div>'
    return '<div class="dica-box" style="color:#475569;">Sinalize para receber feedback!</div>'


# Renderização inicial dos painéis
sign_placeholder.markdown(
    render_sign_panel(st.session_state["last_sign"], st.session_state["last_confidence"]),
    unsafe_allow_html=True
)
chat_placeholder.markdown(
    render_chat(st.session_state["conversation_log"]),
    unsafe_allow_html=True
)
dica_placeholder.markdown(render_dica(), unsafe_allow_html=True)

fps_placeholder.markdown(
    '<div class="metric-card"><div class="metric-label">FPS</div>'
    f'<div class="metric-value accent-cyan">{st.session_state["fps"]}</div></div>',
    unsafe_allow_html=True
)
conf_placeholder.markdown(
    '<div class="metric-card"><div class="metric-label">Confiança</div>'
    f'<div class="metric-value accent-green">{st.session_state["last_confidence"]*100:.0f}%</div></div>',
    unsafe_allow_html=True
)
buf_pct_placeholder.markdown(
    '<div class="metric-card"><div class="metric-label">Buffer</div>'
    f'<div class="metric-value accent-purple">{int(st.session_state["buffer_progress"]*100)}%</div></div>',
    unsafe_allow_html=True
)

# ─── Loop de captura ─────────────────────────────────────────────────────────
if st.session_state["camera_active"]:
    from src.detection.hand_detector import HandDetector
    from src.dialog.intent_mapper import IntentMapper
    from src.utils.helpers import get_sign_emoji

    detector = HandDetector()
    mapper = IntentMapper()

    # Carrega predictor somente se modelo existir
    predictor = None
    if check_model_available():
        try:
            from src.model.predictor import Predictor
            predictor = Predictor(threshold=st.session_state["threshold"])
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Não foi possível acessar a webcam. Verifique a conexão.")
        st.session_state["camera_active"] = False
        st.stop()

    prev_time = time.time()
    last_prediction_time = time.time()
    PREDICTION_COOLDOWN = 1.5  # segundos entre predições

    try:
        while st.session_state["camera_active"]:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame não recebido da câmera.")
                break

            frame = cv2.flip(frame, 1)

            # Detecção de mãos
            results = detector.detect(frame)
            landmarks = detector.extract_landmarks(frame)
            detector.draw_landmarks(frame, results)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # HUD no frame
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 55), (10, 10, 20), -1)

            if detector.has_hands(results):
                cv2.putText(frame, "MAO DETECTADA", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)
            else:
                cv2.putText(frame, "Aguardando mao...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 120), 2)

            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

            # Predição com LSTM
            if predictor:
                predictor.add_frame(landmarks)
                buf_progress = predictor.buffer_progress
                st.session_state["buffer_progress"] = buf_progress

                # Barra de progresso no frame
                bar_w = int(buf_progress * (w - 20))
                cv2.rectangle(frame, (10, h - 20), (w - 10, h - 10), (40, 40, 60), -1)
                cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 10),
                              (100, 80, 220), -1)

                if predictor.is_ready():
                    now = time.time()
                    if now - last_prediction_time >= PREDICTION_COOLDOWN:
                        result = predictor.predict()
                        if result:
                            sign, confidence = result
                            last_prediction_time = now
                            st.session_state["last_sign"] = sign
                            st.session_state["last_confidence"] = confidence
                            st.session_state["total_detections"] += 1

                            # Resposta do sistema
                            response = mapper.get_response(sign)
                            st.session_state["last_response"] = response

                            log_entry = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "sinal": sign,
                                "resposta": response["resposta"],
                                "emoji": response.get("emoji", "🤙"),
                                "dica": response.get("dica", ""),
                            }
                            st.session_state["conversation_log"].append(log_entry)

                            # Atualiza UI
                            sign_placeholder.markdown(
                                render_sign_panel(sign, confidence),
                                unsafe_allow_html=True
                            )
                            chat_placeholder.markdown(
                                render_chat(st.session_state["conversation_log"]),
                                unsafe_allow_html=True
                            )
                            dica_placeholder.markdown(
                                render_dica(response.get("dica", "")),
                                unsafe_allow_html=True
                            )

                            # Overlay no frame
                            cv2.putText(frame, f"{sign} ({confidence*100:.0f}%)",
                                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (100, 220, 255), 2)

            # Exibir frame no Streamlit (BGR → RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Atualizar métricas
            fps_placeholder.markdown(
                '<div class="metric-card"><div class="metric-label">FPS</div>'
                f'<div class="metric-value accent-cyan">{fps:.0f}</div></div>',
                unsafe_allow_html=True
            )
            conf_placeholder.markdown(
                '<div class="metric-card"><div class="metric-label">Confiança</div>'
                f'<div class="metric-value accent-green">'
                f'{st.session_state["last_confidence"]*100:.0f}%</div></div>',
                unsafe_allow_html=True
            )
            buf_pct_placeholder.markdown(
                '<div class="metric-card"><div class="metric-label">Buffer</div>'
                f'<div class="metric-value accent-purple">'
                f'{int(st.session_state.get("buffer_progress", 0)*100)}%</div></div>',
                unsafe_allow_html=True
            )

            # ~30 FPS máximo
            time.sleep(0.033)

    finally:
        cap.release()
        detector.close()
        st.session_state["camera_active"] = False
