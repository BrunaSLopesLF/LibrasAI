# 🤟 LibrasAI — Assistente Inteligente de Conversação em Libras

Sistema de IA em tempo real para prática e aprendizado de Libras utilizando visão computacional (MediaPipe) e redes neurais recorrentes (LSTM).

---

## 📋 Pré-requisitos

- Python 3.9–3.11
- Webcam funcional
- ~2 GB de espaço em disco

---

## 🚀 Instalação

```bash
# 1. Clone ou acesse o diretório do projeto
cd libras-ai

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 🎯 Fluxo de Uso

### Passo 1 — Coletar dados de treinamento

```bash
python -m src.capture.collector
```

- O sistema pedirá o nome do sinal e iniciará a gravação
- Grave **30 sequências** de cada sinal
- Os dados são salvos em `data/raw/<SINAL>/`

### Passo 2 — Treinar o modelo

```bash
python -m src.model.trainer
```

- Treinamento automático com validação 80/20
- Modelo salvo em `models/libras_lstm.h5`
- Gráficos de accuracy/loss gerados em `models/`

### Passo 3 — Executar a aplicação

```bash
streamlit run app.py
```

- Acesse `http://localhost:8501` no navegador
- Clique em **Iniciar Câmera** e comece a sinalizar!

---

## 🤟 Sinais Suportados (MVP)

| Sinal | Descrição |
|-------|-----------|
| OI | Cumprimento |
| OBRIGADO | Agradecimento |
| BOM DIA | Saudação matinal |
| SIM | Afirmação |
| NÃO | Negação |
| AJUDA | Pedido de auxílio |
| EU | Pronome pessoal |
| VOCÊ | Pronome pessoal |
| APRENDER | Verbo/conceito |
| LIBRAS | Nome da língua |

---

## 🏗️ Arquitetura

```
Webcam → OpenCV → MediaPipe (Hands) → Landmarks (126 features)
       → Buffer 30 frames → LSTM → Softmax → Sinal
       → Intent Mapper → Resposta → Streamlit UI
```

---

## 📁 Estrutura do Projeto

```
libras-ai/
├── app.py                    # Interface principal (Streamlit)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                  # Amostras coletadas por sinal
│   └── processed/            # Arrays numpy prontos para treino
├── src/
│   ├── capture/collector.py  # Coleta de amostras
│   ├── detection/hand_detector.py  # Wrapper MediaPipe
│   ├── model/
│   │   ├── lstm_model.py     # Arquitetura LSTM
│   │   ├── trainer.py        # Pipeline de treinamento
│   │   └── predictor.py      # Inferência em tempo real
│   ├── dialog/intent_mapper.py  # Sinal → Resposta
│   └── utils/helpers.py      # Utilitários
├── models/                   # Modelos treinados (.h5)
└── tests/                    # Testes automatizados
```

---

## 🧪 Executar Testes

```bash
pytest tests/ -v
```

---

## 📊 Métricas de Qualidade

- Acurácia mínima: **80%**
- Latência máxima: **< 1 segundo**
- Threshold de confiança: **70%**

---

## 🚀 Roadmap

- [ ] CNN + LSTM para maior precisão
- [ ] Reconhecimento de expressões faciais
- [ ] Avatar 3D comunicando a resposta
- [ ] Integração com IA conversacional
- [ ] App mobile

---

## 📄 Licença

MIT License — Desenvolvido para fins educacionais e de acessibilidade.
