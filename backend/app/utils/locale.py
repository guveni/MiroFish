import json
import os
import threading
from flask import request, has_request_context

_thread_local = threading.local()

_locales_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'locales')

# Load language registry
with open(os.path.join(_locales_dir, 'languages.json'), 'r', encoding='utf-8') as f:
    _languages = json.load(f)

# Load translation files
_translations = {}
for filename in os.listdir(_locales_dir):
    if filename.endswith('.json') and filename != 'languages.json':
        locale_name = filename[:-5]
        with open(os.path.join(_locales_dir, filename), 'r', encoding='utf-8') as f:
            _translations[locale_name] = json.load(f)

_DEFAULT_LOCALE = 'en'


def set_locale(locale: str):
    """Set locale for current thread (e.g. background workers)."""
    _thread_local.locale = locale


def get_locale() -> str:
    if has_request_context():
        raw = (request.headers.get('Accept-Language') or '').strip().split(',')[0].strip()
        if not raw:
            raw = _DEFAULT_LOCALE
        base = raw.split('-')[0] if raw else _DEFAULT_LOCALE
        # Match en, en-US, etc. against loaded bundles
        if raw in _translations:
            return raw
        if base in _translations:
            return base
        return _DEFAULT_LOCALE if _DEFAULT_LOCALE in _translations else next(iter(_translations), _DEFAULT_LOCALE)
    return getattr(_thread_local, 'locale', _DEFAULT_LOCALE)


def t(key: str, **kwargs) -> str:
    locale = get_locale()
    fallback_chain = []
    if locale != _DEFAULT_LOCALE:
        fallback_chain.append(_DEFAULT_LOCALE)
    fallback_chain.extend([k for k in _translations if k not in fallback_chain])

    def _resolve(loc: str) -> str | None:
        messages = _translations.get(loc)
        if not messages:
            return None
        value: object = messages
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        if isinstance(value, str):
            out = value
            if kwargs:
                for k, v in kwargs.items():
                    out = out.replace(f'{{{k}}}', str(v))
            return out
        return None

    for loc in [locale] + fallback_chain:
        got = _resolve(loc)
        if got is not None:
            return got
    return key


def get_language_instruction() -> str:
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get(_DEFAULT_LOCALE, {}))
    return lang_config.get('llmInstruction', 'Please respond in English.')
