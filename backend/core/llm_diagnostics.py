"""Allowlisted provider diagnostics; never echo exception text, prompts or keys."""
import logging

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic['message'])


def provider_diagnostic(provider, error):
    status = getattr(error, 'status_code', None)
    if not isinstance(status, int):
        status = None
    # Inspect raw details for classification only. Never return or log this text.
    text = (type(error).__name__ + ' ' + str(error)).lower()
    if 'model_decommissioned' in text or 'model_not_found' in text or status == 404:
        category, detail = 'model_unavailable', 'The model is unavailable. Check the configured model ID and account access.'
    elif status == 401 or 'authentication' in text or 'invalid_api_key' in text:
        category, detail = 'authentication', 'Authentication failed. Check the provider API key locally.'
    elif status == 403 or 'permissiondenied' in text:
        category, detail = 'permission_denied', 'The account cannot access this model or operation.'
    elif status == 429 or '429' in text or 'rate_limit' in text:
        category, detail = 'rate_limit', 'Rate or quota limit reached. Check provider limits, wait, or configure a fallback provider.'
    elif 'context_length' in text or 'context window' in text or 'maximum context' in text or status == 413:
        category, detail = 'context_limit', 'The request exceeds the provider context/token limit. Reduce the context size.'
    elif status == 400 or status == 422 or 'badrequest' in text:
        category, detail = 'invalid_request', 'The provider rejected the request. Check model support for reasoning/output parameters and token limits.'
    elif 'timeout' in text or 'timed out' in text:
        category, detail = 'timeout', 'The provider timed out. Retry or configure another provider.'
    elif status is not None and status >= 500:
        category, detail = 'provider_unavailable', 'The provider returned a server error. Retry or configure another provider.'
    elif 'connection' in text or 'network' in text:
        category, detail = 'connection', 'The provider could not be reached. Check network/proxy access.'
    elif 'empty response' in text:
        category, detail = 'empty_response', 'The model returned no answer text. Check reasoning settings and the output token budget.'
    elif isinstance(error, ValueError):
        category, detail = 'invalid_response', 'The response could not be used as structured output. Retry or choose another model.'
    else:
        category, detail = 'provider_error', 'The provider request failed. Check the configured model and provider dashboard.'
    # Provider labels originate from our configured clients, not provider error bodies.
    name = provider if provider in {'Groq', 'Gemini', 'xAI', 'OpenAI', 'configured client', 'AI service'} else 'AI service'
    message = f"{name}: {category}" + (f" (HTTP {status})" if status else '') + f". {detail}"
    return {'provider': name, 'category': category, 'http_status': status, 'message': message}


def record_failure(diagnostic):
    logger.warning('%s', diagnostic['message'])
