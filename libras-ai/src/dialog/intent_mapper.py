"""
intent_mapper.py
Mapeamento de sinais em Libras para intenções e respostas do sistema.
Suporte a histórico de conversa e respostas contextuais.
"""

from datetime import datetime
from typing import Optional
from collections import deque


# ─── Base de Conhecimento ───────────────────────────────────────────────────

INTENTS = {
    "OI": {
        "intencao": "cumprimento",
        "resposta": "Olá! Que bom te ver por aqui! 👋",
        "dica": "Você fez o sinal de **OI** corretamente!",
        "emoji": "👋",
    },
    "OBRIGADO": {
        "intencao": "agradecimento",
        "resposta": "De nada! Fico feliz em ajudar no seu aprendizado! 😊",
        "dica": "Perfeito! O sinal de **OBRIGADO** reconhecido!",
        "emoji": "🙏",
    },
    "BOM_DIA": {
        "intencao": "saudacao_matinal",
        "resposta": "Bom dia! Pronto para mais uma aula de Libras? ☀️",
        "dica": "Ótimo! Sinal de **BOM DIA** identificado!",
        "emoji": "☀️",
    },
    "SIM": {
        "intencao": "afirmacao",
        "resposta": "Ótimo! Concordo! Vamos continuar praticando! ✅",
        "dica": "Correto! O sinal de **SIM** está perfeito!",
        "emoji": "✅",
    },
    "NAO": {
        "intencao": "negacao",
        "resposta": "Tudo bem! Sem problemas. Podemos tentar de novo! 🔄",
        "dica": "Muito bem! Sinal de **NÃO** reconhecido!",
        "emoji": "❌",
    },
    "AJUDA": {
        "intencao": "pedido_auxilio",
        "resposta": "Claro! Estou aqui para ajudar. O que você precisa? 🤝",
        "dica": "Perfeito! Sinal de **AJUDA** detectado!",
        "emoji": "🤝",
    },
    "EU": {
        "intencao": "pronome_eu",
        "resposta": "Você! Gosto de aprender sobre você. Me conte mais! 😄",
        "dica": "Ótimo! O sinal de **EU** está correto!",
        "emoji": "👤",
    },
    "VOCE": {
        "intencao": "pronome_voce",
        "resposta": "Eu? Sou seu assistente de Libras! Posso te ajudar muito! 🤖",
        "dica": "Correto! Sinal de **VOCÊ** reconhecido!",
        "emoji": "👉",
    },
    "APRENDER": {
        "intencao": "aprendizado",
        "resposta": "Aprender Libras é incrível! Você está no caminho certo! 📚",
        "dica": "Excelente! Sinal de **APRENDER** identificado!",
        "emoji": "📚",
    },
    "LIBRAS": {
        "intencao": "lingua_sinais",
        "resposta": "Libras é uma língua rica e expressiva! Continue praticando! 🤟",
        "dica": "Perfeito! Sinal de **LIBRAS** detectado!",
        "emoji": "🤟",
    },
}

# ─── Respostas contextuais (baseadas em sequência de sinais) ─────────────────

CONTEXT_RESPONSES = {
    ("OI", "VOCE"): "Oi! Sim, sou eu, seu assistente LibrasAI! 🤖👋",
    ("EU", "APRENDER"): "Que ótimo que você quer aprender! Estamos começando bem! 🌟",
    ("EU", "APRENDER", "LIBRAS"): "Perfeito! 'EU APRENDER LIBRAS' — frase completa! Excelente! 🏆",
    ("VOCE", "AJUDA"): "Claro! Estou sempre aqui para te ajudar a aprender Libras! 💪",
    ("OI", "BOM_DIA"): "Olá, bom dia! Que energia boa para aprender! ☀️👋",
    ("SIM", "OBRIGADO"): "De nada! Fico feliz que gostou! Continue assim! 😊✅",
    ("EU", "VOCE"): "Eu e você aprendendo Libras juntos! Que legal! 👤👉",
}

# ─── Resposta de fallback ─────────────────────────────────────────────────────

FALLBACK_RESPONSE = {
    "resposta": "Hmm, não entendi esse sinal ainda. Tente novamente! 🤔",
    "dica": "O sinal não foi reconhecido com confiança suficiente.",
    "emoji": "🤔",
}

UNKNOWN_RESPONSE = {
    "resposta": "Esse sinal ainda não está no meu dicionário. Mas estou aprendendo! 📖",
    "dica": "Sinal não mapeado.",
    "emoji": "📖",
}


class IntentMapper:
    """
    Mapeia sinais reconhecidos para intenções e gera respostas contextuais.
    Mantém histórico da conversa para respostas mais naturais.
    """

    def __init__(self, history_size: int = 5):
        """
        Args:
            history_size: Número de sinais a manter no histórico.
        """
        self.history: deque = deque(maxlen=history_size)
        self.conversation_log: list = []

    def get_response(self, sign: str) -> dict:
        """
        Gera resposta para um sinal reconhecido.

        Verifica primeiro o contexto (últimos sinais) para resposta
        mais personalizada. Caso contrário, usa a resposta padrão.

        Args:
            sign: Nome do sinal reconhecido (ex: 'OI', 'OBRIGADO').

        Returns:
            Dicionário com: resposta, dica, emoji, intencao.
        """
        sign_upper = sign.upper()
        self.history.append(sign_upper)

        # Verifica resposta contextual (últimos 3 e 2 sinais)
        for window in [3, 2]:
            if len(self.history) >= window:
                context_key = tuple(list(self.history)[-window:])
                if context_key in CONTEXT_RESPONSES:
                    response = {
                        "resposta": CONTEXT_RESPONSES[context_key],
                        "dica": f"Sequência detectada: {' → '.join(context_key)}",
                        "emoji": "🌟",
                        "intencao": "contextual",
                        "sinal": sign_upper,
                        "contextual": True,
                    }
                    self._log(sign_upper, response)
                    return response

        # Resposta padrão do sinal
        if sign_upper in INTENTS:
            intent = INTENTS[sign_upper]
            response = {
                **intent,
                "sinal": sign_upper,
                "contextual": False,
            }
        else:
            response = {
                **UNKNOWN_RESPONSE,
                "sinal": sign_upper,
                "intencao": "desconhecido",
                "contextual": False,
            }

        self._log(sign_upper, response)
        return response

    def _log(self, sign: str, response: dict):
        """Registra interação no log da conversa."""
        self.conversation_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sinal": sign,
            "resposta": response.get("resposta", ""),
            "emoji": response.get("emoji", ""),
        })

    def get_history_display(self) -> list:
        """
        Retorna o histórico formatado para exibição na UI.

        Returns:
            Lista de dicionários com timestamp, sinal, resposta e emoji.
        """
        return list(reversed(self.conversation_log))

    def get_recent_signs(self) -> list:
        """Retorna os sinais recentes do buffer de histórico."""
        return list(self.history)

    def reset(self):
        """Limpa histórico e log de conversa."""
        self.history.clear()
        self.conversation_log.clear()

    @staticmethod
    def list_known_signs() -> list:
        """Retorna a lista de sinais suportados."""
        return list(INTENTS.keys())
