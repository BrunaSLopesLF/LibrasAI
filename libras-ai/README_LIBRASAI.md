# LibrasAI 🖐️🤖

O **LibrasAI** é um ecossistema inteligente de comunicação e reconhecimento de Língua Brasileira de Sinais (Libras) em tempo real. O sistema utiliza técnicas avançadas de Visão Computacional e Aprendizado Profundo para capturar movimentos da webcam e traduzi-los instantaneamente em texto, integrando uma interface de usuário interativa a um servidor de inferência de alta performance.

---

## 🏗️ Arquitetura do Sistema

O projeto foi estruturado seguindo os padrões de engenharia de software back-end, dividido em três componentes principais:

1. **Pipeline de Visão Computacional (MediaPipe):** Captura o fluxo de vídeo da webcam e extrai os pontos de articulação espaciais (*landmarks*). O sistema aplica um fatiamento (*slicing*) estrito para isolar as **42 features tridimensionais** essenciais da mão principal, otimizando o processamento.
2. **Back-end e Servidor de Inferência (FastAPI & WebSockets):** Um servidor assíncrono de alta performance que gerencia a comunicação bidirecional em tempo real via WebSockets, recebendo as sequências de frames e processando a inferência sem gargalos.
3. **Rede Neural Recorrente (TensorFlow/Keras):** Modelo baseado em camadas **LSTM (Long Short-Term Memory)** integradas a uma **Camada de Atenção Customizada (`AttentionLayer`)**, projetada para identificar a relevância temporal de janelas de 30 frames, permitindo alta acurácia na diferenciação de sinais complexos e pronomes (como *EU* e *VOCÊ*).
4. **Front-end Interativo (Streamlit):** Interface de usuário responsiva que exibe o feedback visual das predições, métricas de desempenho e o histórico de conversação gerado.

---

## 📊 Resultados e Desempenho

O modelo foi treinado localmente utilizando um dataset balanceado de **50 sequências por sinal** para 10 classes oficiais.
* **Acurácia Geral:** 73% nos dados de validação.
* **Destaques:** Sinais como *OBRIGADO* atingiram 100% de F1-Score, e pronomes de alta complexidade como *VOCÊ* alcançaram 89% de precisão.
* **Acessibilidade:** Projetado para rodar o pipeline completo localmente de forma eficiente em CPU (com taxas de 5 a 11 FPS), eliminando a dependência de GPUs dedicadas de alto custo.

---

## 🚀 Como Executar o Projeto

1. **Ative o ambiente virtual:**
   ```bash
   source venv/bin/activate

2. **Inicie o Back-end (FastAPI):**

  bash
	export PYTHONPATH=$PYTHONPATH:.
	python server.py
3. **Inicie o Front-end (Streamlit):**

  bash
	streamlit run app.py
