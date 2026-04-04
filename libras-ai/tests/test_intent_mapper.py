"""
test_intent_mapper.py
Testes unitários para o módulo IntentMapper.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dialog.intent_mapper import IntentMapper, INTENTS, CONTEXT_RESPONSES


@pytest.fixture
def mapper():
    """IntentMapper com estado limpo para cada teste."""
    return IntentMapper()


class TestKnownSigns:
    def test_list_known_signs_returns_list(self):
        signs = IntentMapper.list_known_signs()
        assert isinstance(signs, list)
        assert len(signs) == 10

    def test_all_expected_signs_present(self):
        signs = IntentMapper.list_known_signs()
        expected = {"OI", "OBRIGADO", "BOM_DIA", "SIM", "NAO",
                    "AJUDA", "EU", "VOCE", "APRENDER", "LIBRAS"}
        assert expected == set(signs)


class TestGetResponse:
    def test_known_sign_returns_response(self, mapper):
        resp = mapper.get_response("OI")
        assert "resposta" in resp
        assert len(resp["resposta"]) > 0

    def test_response_has_required_fields(self, mapper):
        resp = mapper.get_response("OBRIGADO")
        for field in ["resposta", "dica", "emoji", "sinal"]:
            assert field in resp, f"Campo '{field}' ausente na resposta"

    def test_sign_stored_in_response(self, mapper):
        resp = mapper.get_response("SIM")
        assert resp["sinal"] == "SIM"

    def test_lowercase_sign_accepted(self, mapper):
        """Sinais em minúsculo devem ser normalizados."""
        resp = mapper.get_response("oi")
        assert resp["sinal"] == "OI"

    def test_unknown_sign_returns_fallback(self, mapper):
        resp = mapper.get_response("SIGN_INEXISTENTE_XYZ")
        assert "resposta" in resp
        assert resp["sinal"] == "SIGN_INEXISTENTE_XYZ"

    def test_all_known_signs_return_response(self, mapper):
        """Todos os sinais mapeados devem retornar resposta válida."""
        for sign in IntentMapper.list_known_signs():
            resp = mapper.get_response(sign)
            assert resp["resposta"], f"Sinal '{sign}' retornou resposta vazia"


class TestContextualResponses:
    def test_sequential_signs_trigger_context(self, mapper):
        """Sequência EU → APRENDER → LIBRAS deve gerar resposta contextual."""
        mapper.get_response("EU")
        mapper.get_response("APRENDER")
        resp = mapper.get_response("LIBRAS")
        assert resp.get("contextual") is True

    def test_context_response_has_message(self, mapper):
        mapper.get_response("EU")
        resp = mapper.get_response("APRENDER")
        # EU → APRENDER está no CONTEXT_RESPONSES
        assert "resposta" in resp
        assert len(resp["resposta"]) > 0

    def test_non_contextual_sign_is_flagged(self, mapper):
        """Sinal sem contexto especial deve ter contextual=False."""
        resp = mapper.get_response("LIBRAS")
        assert resp.get("contextual") is False


class TestHistory:
    def test_history_updates(self, mapper):
        mapper.get_response("OI")
        mapper.get_response("SIM")
        history = mapper.get_recent_signs()
        assert "OI" in history
        assert "SIM" in history

    def test_conversation_log_grows(self, mapper):
        mapper.get_response("OI")
        mapper.get_response("OBRIGADO")
        log = mapper.get_history_display()
        assert len(log) >= 2

    def test_reset_clears_state(self, mapper):
        mapper.get_response("OI")
        mapper.get_response("SIM")
        mapper.reset()
        assert len(mapper.get_recent_signs()) == 0
        assert len(mapper.get_history_display()) == 0

    def test_history_display_is_reversed(self, mapper):
        """Histórico exibido deve ter o mais recente primeiro."""
        mapper.get_response("OI")
        mapper.get_response("LIBRAS")
        log = mapper.get_history_display()
        if len(log) >= 2:
            assert log[0]["sinal"] == "LIBRAS"
            assert log[1]["sinal"] == "OI"
