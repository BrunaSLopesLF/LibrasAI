const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const signPanel = document.getElementById('sign-panel');
const chatContainer = document.getElementById('chat-container');
const bufferBar = document.getElementById('buffer-bar');
const wsStatus = document.getElementById('ws-status');
const metricTotal = document.getElementById('metric-total');
const metricHands = document.getElementById('metric-hands');
const statusOverlay = document.getElementById('status-overlay');

// Nomes Elementos Novos
const btnWebcam = document.getElementById('btn-webcam');
const btnVideo = document.getElementById('btn-video');
const videoUpload = document.getElementById('video-upload');
const currentContextText = document.getElementById('current-context-text');
const btnClearContext = document.getElementById('btn-clear-context');
const aiChatInput = document.getElementById('ai-chat-input');
const btnSendMessage = document.getElementById('btn-send-message');

let ws = null;
let currentSequence = [];
const SEQUENCE_LENGTH = 30;
let totalPredictions = 0;
let currentContext = [];
let activeSource = 'webcam';
let videoRAF = null;

function connectWebSocket() {
    // Para simplificar, tenta achar a URL dinamicamente ou localhost
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Se estiver rodando sem servidor HTTP (file://), força localhost:8000
    const host = (window.location.protocol === 'file:') ? 'localhost:8000' : window.location.host;
    
    const wsUrl = `${wsProtocol}//${host}/ws/predict`;
    console.log(`Conectando WS em ${wsUrl}`);
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        wsStatus.innerHTML = '🟢 Conectado';
        wsStatus.className = 'metric-value accent-green';
    };

    ws.onclose = () => {
        wsStatus.innerHTML = '🔴 Desconectado';
        wsStatus.className = 'metric-value';
        setTimeout(connectWebSocket, 3000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handlePredictionOutput(data);
    };
}

function handlePredictionOutput(data) {
    if(!data) return;
    
    totalPredictions++;
    metricTotal.innerText = totalPredictions;

    // Atualiza Painel Secundário
    signPanel.innerHTML = `
        <div style="font-size:2.5rem;">🤟</div>
        <div class="sign-name">${data.prediction.replace('_', ' ')}</div>
        <div class="sign-confidence">Confiança: ${(data.confidence * 100).toFixed(1)}%</div>
    `;

    // Atualiza Chat se houver resposta válida
    if (data.prediction !== "INDEFINIDO") {
        const signText = data.prediction.replace('_', ' ');
        if(currentContext.length === 0 || currentContext[currentContext.length-1] !== signText) {
             currentContext.push(signText);
             currentContextText.innerText = currentContext.join(' ');
        }

        const msgHtml = `
            <div class="chat-message">
                <div class="chat-sinal">${signText}</div>
                <div class="chat-text">${data.response || '...'}</div>
            </div>
        `;
        chatContainer.insertAdjacentHTML('beforeend', msgHtml);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Configuração do MediaPipe
statusOverlay.style.display = 'block';

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(onResults);

function onResults(results) {
    statusOverlay.style.display = 'none';

    // Ajustar o canvas para o tamanho do video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        metricHands.innerHTML = 'Sim';
        metricHands.className = 'metric-value accent-cyan';

        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
            
            // Extrair coordenadas X, Y (21 * 2 = 42 coordenadas)
            // Caso seu modelo tenha treinado com Z, basta incluir lm.z
            let coords = [];
            landmarks.forEach(lm => {
                coords.push(lm.x, lm.y); 
            });

            currentSequence.push(coords);

            // Interface do Buffer
            const pct = (currentSequence.length / SEQUENCE_LENGTH) * 100;
            bufferBar.style.width = pct + '%';

            // Envia apenas quando chegar no final da sequência (janela deslizante simples)
            if (currentSequence.length === SEQUENCE_LENGTH) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ 
                        sequence: currentSequence, 
                        timestamp: Date.now() 
                    }));
                }
                // Limpa o buffer para o próximo
                currentSequence = [];
                bufferBar.style.width = '0%';
            }
        }
    } else {
        metricHands.innerHTML = 'Não';
        metricHands.className = 'metric-value accent-purple';
        // Opcional: Esvaziar buffer se a mão sumir
        // currentSequence = []; 
        // bufferBar.style.width = '0%';
    }
    
    canvasCtx.restore();
}

// Função para disparar frames do vídeo gravado
async function processVideoFrame() {
    if (activeSource !== 'video' || videoElement.paused || videoElement.ended) return;
    await hands.send({image: videoElement});
    videoRAF = requestAnimationFrame(processVideoFrame);
}

videoElement.addEventListener('play', () => {
    if (activeSource === 'video') {
         processVideoFrame();
    }
});

// Eventos de Seleção de Fonte
btnWebcam.addEventListener('click', () => {
    if (activeSource === 'webcam') return;
    activeSource = 'webcam';
    btnWebcam.classList.add('active');
    btnVideo.classList.remove('active');
    videoElement.src = '';
    videoElement.removeAttribute('controls');
    videoElement.style.transform = 'scaleX(-1)';
    canvasElement.style.transform = 'scaleX(-1)';
    camera.start();
});

btnVideo.addEventListener('click', () => {
    activeSource = 'video';
    btnVideo.classList.add('active');
    btnWebcam.classList.remove('active');
    camera.stop(); 
    videoElement.style.transform = 'none'; 
    canvasElement.style.transform = 'none';
    videoUpload.click();
});

videoUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        videoElement.src = url;
        videoElement.controls = true;
        videoElement.play();
    }
});

// Eventos do Contexto e Chat IA
btnClearContext.addEventListener('click', () => {
    currentContext = [];
    currentContextText.innerText = 'Nenhum sinal detectado ainda...';
});

btnSendMessage.addEventListener('click', () => {
    const text = aiChatInput.value.trim();
    if (!text) return;
    
    const msgHtml = `
        <div class="chat-message" style="border-left: 4px solid var(--accent-blue);">
            <div class="chat-sinal">Você (Chat Livre)</div>
            <div class="chat-text">${text}</div>
        </div>
    `;
    chatContainer.insertAdjacentHTML('beforeend', msgHtml);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'chat_message',
            text: text,
            context: currentContext.join(' ')
        }));
    }
    
    aiChatInput.value = '';
});

aiChatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') btnSendMessage.click();
});

// Inicializar a Câmera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    if (activeSource === 'webcam') {
        await hands.send({image: videoElement});
    }
  },
  width: 640,
  height: 480
});

// Start loop
camera.start();
connectWebSocket();
