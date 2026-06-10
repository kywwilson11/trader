"""Critical-event notifications for an unattended trading system.

The bots run 24/7 on a Jetson in a home — when the circuit breaker trips or
a flatten fails at 15:55 ET, a log line nobody is watching is not enough.

Channels (set one or both in .env):
  TRADER_WEBHOOK_URL       Discord-compatible webhook (also works with
                           Slack via /slack suffix, ntfy.sh JSON, etc.)
  TRADER_TELEGRAM_BOT_TOKEN + TRADER_TELEGRAM_CHAT_ID

Design rules:
  - NEVER raises and NEVER blocks a trading path: sends happen on a
    daemon thread with a short timeout.
  - Deduped: the same dedupe_key alerts at most once per 10 minutes
    (a crash-looping cycle must not send 120 messages an hour).
  - No-op when no channel is configured.
"""

import json
import os
import threading
import time
import urllib.request

from log_config import get_logger

logger = get_logger(__name__)

_DEDUPE_WINDOW_SEC = 600
_last_sent: dict[str, float] = {}
_lock = threading.Lock()

LEVEL_EMOJI = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=10)


def _send(message: str, level: str):
    text = f"{LEVEL_EMOJI.get(level, '')} [trader/{level}] {message}"[:1900]
    webhook = os.getenv('TRADER_WEBHOOK_URL')
    if webhook:
        try:
            _post(webhook, {'content': text})
        except Exception as e:
            logger.debug('[NOTIFY] webhook failed: %s', e)
    token = os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
    chat = os.getenv('TRADER_TELEGRAM_CHAT_ID')
    if token and chat:
        try:
            _post(f"https://api.telegram.org/bot{token}/sendMessage",
                  {'chat_id': chat, 'text': text})
        except Exception as e:
            logger.debug('[NOTIFY] telegram failed: %s', e)


_hb_last: dict[str, float] = {}


def ping_heartbeat(name: str):
    """Dead-man's-switch ping (healthchecks.io-style) + local mtime file.

    Alerting on EVENTS can't catch power loss, network death, or a hung
    process — only a missing heartbeat can. Configure per-loop URLs
    (TRADER_HEALTHCHECK_URL_CRYPTO / _STOCK) or one shared
    TRADER_HEALTHCHECK_URL; the local `{name}_heartbeat` file serves any
    on-device watchdog. Rate-limited to one ping/minute; never blocks.
    """
    try:
        now = time.monotonic()
        with _lock:
            if now - _hb_last.get(name, -1e9) < 60:
                return
            _hb_last[name] = now
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'{name}_heartbeat')
        with open(path, 'w') as f:
            f.write(str(time.time()))
        url = (os.getenv(f'TRADER_HEALTHCHECK_URL_{name.upper()}')
               or os.getenv('TRADER_HEALTHCHECK_URL'))
        if url:
            def _ping():
                try:
                    urllib.request.urlopen(url, timeout=10)
                except Exception:
                    pass
            threading.Thread(target=_ping, daemon=True, name='heartbeat').start()
    except Exception:
        pass


def notify(message: str, level: str = 'warning', dedupe_key: str | None = None):
    """Fire-and-forget alert. Safe to call from any trading path."""
    try:
        if not (os.getenv('TRADER_WEBHOOK_URL')
                or os.getenv('TRADER_TELEGRAM_BOT_TOKEN')):
            return
        key = dedupe_key or message[:80]
        now = time.monotonic()
        with _lock:
            if now - _last_sent.get(key, -1e9) < _DEDUPE_WINDOW_SEC:
                return
            _last_sent[key] = now
            if len(_last_sent) > 200:
                cutoff = now - _DEDUPE_WINDOW_SEC
                for k in [k for k, t in _last_sent.items() if t < cutoff]:
                    del _last_sent[k]
        threading.Thread(target=_send, args=(message, level),
                         daemon=True, name='notify').start()
    except Exception:
        pass  # alerting must never break trading
