import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import config
from app_api import LegalAPIRequestHandler
from core.legal_reviewer import LegalReviewer
from core.document_loader import LoadedDocument, PageData
from core.llm_diagnostics import LLMProviderError, provider_diagnostic
import core.legal_reviewer as reviewer_module


class Client:
    def __init__(self, error=None, content='OK'):
        self.error, self.content = error, content
    def bind(self, **kwargs):
        return self
    def invoke(self, messages):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def reviewer():
    return LegalReviewer(LoadedDocument('<memory>', 'test.pdf', 0), None)


def error(status, message='DO-NOT-EXPOSE-KEY OR CONTRACT CONTENT'):
    result = RuntimeError(message)
    result.status_code = status
    return result


@pytest.mark.parametrize('status,category', [(401,'authentication'), (403,'permission_denied'),
    (404,'model_unavailable'), (429,'rate_limit'), (400,'invalid_request'), (413,'context_limit'), (503,'provider_unavailable')])
def test_provider_errors_are_actionable_and_never_echo_sensitive_text(status, category, caplog, monkeypatch):
    r = reviewer()
    r._llm_candidates = [('Groq', Client(error(status)))]
    monkeypatch.setattr(reviewer_module.time, 'sleep', lambda _: None)
    with pytest.raises(LLMProviderError) as failure:
        r._invoke_llm([])
    diagnostic = failure.value.diagnostic
    assert diagnostic['category'] == category
    assert diagnostic['http_status'] == status
    assert str(status) in r.last_llm_diagnostic
    assert 'DO-NOT-EXPOSE' not in json.dumps(diagnostic) + caplog.text + str(failure.value)


def test_rate_limit_reaches_qa_response_as_rate_limit(monkeypatch):
    r = reviewer()
    r._llm_candidates = [('Groq', Client(error(429)))]
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(reviewer_module.time, 'sleep', lambda _: None)
    result = r.answer_query('What is the payment period?')
    assert result['status'] == 'analysis_failed'
    assert result['diagnostic']['category'] == 'rate_limit'


def test_negotiation_exposes_parse_failure_separately(monkeypatch):
    r = reviewer()
    r._llm_candidates = [('Groq', Client(content='not valid JSON'))]
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    result = r.generate_negotiation_strategy('Payment is due in 30 days.')
    assert result['diagnostic']['category'] == 'invalid_response'


def test_empty_obligation_fallback_still_has_failure_metadata(monkeypatch):
    r = reviewer()
    r.document.pages = [PageData(1, "No explicit duties here.", 24, 0)]
    r._llm_candidates = [('Groq', Client(error(401)))]
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    assert r.extract_obligations() == []
    assert r.obligation_status['mode'] == 'rule_based'
    assert r.obligation_status['diagnostic']['category'] == 'authentication'


@pytest.mark.parametrize('exception', [ConnectionAbortedError, ConnectionResetError, BrokenPipeError])
@pytest.mark.parametrize('during_headers', [True, False])
def test_disconnected_client_never_gets_a_second_response(exception, during_headers):
    handler = object.__new__(LegalAPIRequestHandler)
    handler.path = '/api/visuals'
    sent_headers, writes, logs = [], [], []
    def set_headers(*args):
        sent_headers.append(args)
        if during_headers:
            raise exception('client disconnected')
    def write(data):
        writes.append(data)
        raise exception('client disconnected')
    handler._set_headers = set_headers
    handler.wfile = SimpleNamespace(write=write)
    handler.log_message = lambda *args: logs.append(args)
    handler._send_json({'previews': ['image']})
    handler._send_json({'error': 'should not be sent'}, 500)
    assert len(sent_headers) == 1
    assert len(writes) == (0 if during_headers else 1)
    assert handler.close_connection
    assert len(logs) == 1


def test_groq_does_not_force_optional_reasoning_parameters(monkeypatch):
    monkeypatch.setenv('LEGAL_AI_OFFLINE', 'false')
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_PROVIDER_ORDER', 'groq')
    monkeypatch.delenv('GROQ_REASONING_EFFORT', raising=False)
    monkeypatch.delenv('GROQ_INCLUDE_REASONING', raising=False)
    monkeypatch.setattr(config, 'ChatOpenAI', lambda **kw: kw)
    options = config.get_llm_candidates()[0][1]
    assert 'reasoning_effort' not in options
    assert 'extra_body' not in options
    monkeypatch.setenv('GROQ_REASONING_EFFORT', 'none')
    monkeypatch.setenv('GROQ_INCLUDE_REASONING', 'false')
    options = config.get_llm_candidates()[0][1]
    assert options['reasoning_effort'] == 'none'
    assert options['extra_body'] == {'include_reasoning': False}


def test_connection_check_route_reports_safe_failure_without_a_document(monkeypatch):
    handler = object.__new__(LegalAPIRequestHandler)
    handler.path = '/api/llm/check'
    handler._read_json = lambda: {}
    responses = []
    handler._send_json = lambda data, status=200: responses.append((data, status))
    def fail(self, messages, **kwargs):
        assert messages[0].content == 'Reply with the word OK.'
        raise LLMProviderError(provider_diagnostic('Groq', error(401)))
    monkeypatch.setattr(LegalReviewer, '_invoke_llm', fail)
    handler._handle_post()
    assert responses[0][0]['diagnostic']['category'] == 'authentication'
    assert responses[0][0]['status'] == 'failed'


def test_obligation_status_envelope_preserves_legacy_array():
    r = reviewer()
    r.extract_obligations = lambda: []
    r.obligation_status = {'mode':'rule_based', 'diagnostic':{'message':'AI unavailable'}}
    handler = object.__new__(LegalAPIRequestHandler)
    handler._require_reviewer = lambda: r
    responses = []
    handler._send_json = lambda data, status=200: responses.append(data)
    handler.path = '/api/obligations?include_status=1'
    handler._handle_get()
    assert responses[-1]['mode'] == 'rule_based'
    handler.path = '/api/obligations'
    handler._handle_get()
    assert responses[-1] == []
