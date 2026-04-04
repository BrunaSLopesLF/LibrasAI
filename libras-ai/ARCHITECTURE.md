# Arquitetura do Sistema: Libras AI

Este documento detalha o funcionamento e o fluxo de dados arquitetural do sistema **Libras AI**, de forma amigável para o entendimento do público em geral e novos desenvolvedores do projeto.

O fluxo ilustra desde a captura da imagem do usuário até a tradução da linguagem de sinais e posterior interação com um agente de Inteligência Artificial usando o contexto estabelecido.

## Fluxograma do Sistema 

```mermaid
graph TD
    %% Estilos Personalizados
    classDef frontend fill:#3b82f6,stroke:#1e3a8a,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef mediaPipe fill:#10b981,stroke:#064e3b,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef backend fill:#8b5cf6,stroke:#4c1d95,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef ai fill:#f59e0b,stroke:#78350f,stroke-width:2px,color:#fff,rx:8px,ry:8px

    subgraph UserInterface["1. Interface do Usuário (Navegador)"]
        A1["🎥 Câmera Ao Vivo (Webcam)"]:::frontend
        A2["📁 Upload de Vídeo Gravado"]:::frontend
        UI["💬 Tela de Chat e Resultados"]:::frontend
    end

    subgraph ComputerVision["2. Módulo de Visão Computacional (Frontend)"]
        MP["Google MediaPipe Hands<br>(Extração do 'esqueleto' da mão)"]:::mediaPipe
    end

    subgraph ServerBackend["3. Servidor Backend (FastAPI / Servidor Nuvem)"]
        WS["🌐 Servidor WebSocket<br>(Comunicação ultrarrápida bi-direcional)"]:::backend
        LSTM["🧠 Rede Neural Deep Learning<br>(LSTM + Attention)<br>(Decifra qual é o Sinal de Libras)"]:::backend
        LLM["🤖 Agente Inteligência Artificial<br>(Compreende todo o contexto e conversa)"]:::ai
    end

    %% Fluxo de Dados (Caminho Feliz)
    A1 -->|Envia Imagem| MP
    A2 -->|Envia Imagem| MP
    MP -->|Mapeamento 3D das Articulações| WS
    WS -->|Sequência de Movimentos (30 frames)| LSTM
    LSTM -->|Sinal Traduzido (Ex: 'Bom dia')| WS
    WS -->|Atualiza Layout/Contexto| UI
    UI -->|Pergunta do Usuário (Chat Livre)| WS
    WS -->|Contexto de Sinais + Pergunta em Texto| LLM
    LLM -->|Resposta Assistiva/Inteligente| WS
    WS -->|Exibe Mensagem Final| UI
```

## Como Ler Este Fluxograma

O sistema opera de forma contínua e em tempo real, dividido em **três etapas principais**:

1. **Captura Visual (Interface):**
   O usuário decide se usará a câmera do computador para executar os sinais ao vivo ou se fará o envio de um vídeo pré-gravado.

2. **Processamento Leve de Imagem (Visão Computacional no Frontend):**
   Para garantir que dados pesados de vídeo não trafeguem pela internet, usamos a tecnologia **Google MediaPipe** no próprio navegador do usuário. Ela isola e entende apenas a estrutura óssea das mãos (coordenadas espaciaais). Apenas essas coordenadas textuais – que são muito leves – são enviadas ao servidor pela conexão WebSocket.

3. **Inteligência Central (Backend e Deep Learning):**
   O Servidor enxerga o tempo e o movimento contínuo da mão agrupando coordenadas (ex: a cada 30 quadros). Ele passa essa sequência de movimento por nossa **Rede Neural (LSTM)** que tem uma "atenção" treinada (layer de *Attention*) focada em reconhecer precisamente o sinal em Libras que acabou de ser feito.
   A tradução é revelada ao usuário via chat, criando um "Contexto". Então um **Agente de IA (LLM - Large Language Model)** entra em cena, atuando de assistente para quem quiser tirar dúvidas sobre as transcrições usando a caixa de chat livre.
