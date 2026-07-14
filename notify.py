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
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def _send(message: str, level: str):
    # Delivery failures are warnings, not debug: a misconfigured webhook
    # would otherwise swallow every CRITICAL alert invisibly at the default
    # INFO log level. Volume is bounded by notify()'s 10-minute dedupe gate.
    text = f"{LEVEL_EMOJI.get(level, '')} [trader/{level}] {message}"[:1900]
    webhook = os.getenv('TRADER_WEBHOOK_URL')
    if webhook:
        try:
            _post(webhook, {'content': text})
        except Exception as e:
            logger.warning('[NOTIFY] webhook failed: %s', e)
    token = os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
    chat = os.getenv('TRADER_TELEGRAM_CHAT_ID')
    if token and chat:
        try:
            _post(f"https://api.telegram.org/bot{token}/sendMessage",
                  {'chat_id': chat, 'text': text})
        except Exception as e:
            logger.warning('[NOTIFY] telegram failed: %s', e)


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
                    with urllib.request.urlopen(url, timeout=10) as r:
                        r.read()
                except Exception:
                    pass
            threading.Thread(target=_ping, daemon=True, name='heartbeat').start()
    except Exception as e:
        # e.g. SD card read-only/full — the remote healthcheck will still
        # fire, but leave an on-device trace for diagnosis
        logger.debug('[NOTIFY] heartbeat failed: %s', e)


# --- Remote kill switch (Telegram) + halt flag ---
#
# The halt flag is a plain file so EVERY control surface can flip it:
# Telegram /halt, the GUI, or `touch trading_halt.flag` over ssh. Loops
# check it before any new entry; exits and stops keep running.

_HALT_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'trading_halt.flag')
_FLATTEN_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'flatten_request.flag')
_TG_OFFSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'telegram_offset.json')

# Escalation state for poll_telegram_commands: transient poll blips stay at
# debug, but a CONTINUOUSLY failing poll means /halt and /flatten are dead —
# warn (at most hourly) so the operator learns the kill switch is offline.
_tg_fail_since: float | None = None
_tg_last_warn = -1e9
_TG_FAIL_WARN_AFTER_SEC = 600
_TG_FAIL_WARN_EVERY_SEC = 3600


def halt_active() -> bool:
    return os.path.exists(_HALT_FLAG)


def set_halt(reason: str = ''):
    try:
        with open(_HALT_FLAG, 'w') as f:
            f.write(json.dumps({'reason': reason,
                                'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}))
    except OSError:
        pass


def clear_halt():
    try:
        os.remove(_HALT_FLAG)
    except OSError:
        pass


def flatten_requested() -> bool:
    return os.path.exists(_FLATTEN_FLAG)


def request_flatten(reason: str = ''):
    try:
        with open(_FLATTEN_FLAG, 'w') as f:
            f.write(reason)
    except OSError:
        pass


def clear_flatten_request():
    try:
        os.remove(_FLATTEN_FLAG)
    except OSError:
        pass


def poll_telegram_commands() -> list[str]:
    """New commands from the configured Telegram chat (kill switch).

    Uses getUpdates with a persisted offset; ONLY accepts messages whose
    chat id matches TRADER_TELEGRAM_CHAT_ID (anyone on Telegram can
    message a bot — without this check a stranger could halt trading).
    Returns lowercase command strings like '/halt'. Never raises.
    """
    global _tg_fail_since, _tg_last_warn
    token = os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
    chat = os.getenv('TRADER_TELEGRAM_CHAT_ID')
    if not token or not chat:
        return []
    try:
        offset = 0
        try:
            with open(_TG_OFFSET_FILE) as f:
                offset = int(json.load(f).get('offset', 0))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        url = (f"https://api.telegram.org/bot{token}/getUpdates"
               f"?offset={offset + 1}&timeout=0&allowed_updates=%5B%22message%22%5D")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        cmds = []
        max_id = offset
        for upd in data.get('result', []):
            max_id = max(max_id, int(upd.get('update_id', 0)))
            msg = upd.get('message') or {}
            if str((msg.get('chat') or {}).get('id')) != str(chat):
                continue
            text = (msg.get('text') or '').strip().lower()
            if text.startswith('/'):
                cmds.append(text.split('@')[0].split()[0])
        if max_id > offset:
            tmp = _TG_OFFSET_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'offset': max_id}, f)
            os.replace(tmp, _TG_OFFSET_FILE)
        _tg_fail_since = None
        return cmds
    except Exception as e:
        logger.debug('[NOTIFY] telegram poll failed: %s', e)
        now = time.monotonic()
        if _tg_fail_since is None:
            _tg_fail_since = now
        elif (now - _tg_fail_since >= _TG_FAIL_WARN_AFTER_SEC
                and now - _tg_last_warn >= _TG_FAIL_WARN_EVERY_SEC):
            _tg_last_warn = now
            logger.warning('[NOTIFY] telegram polling failing for %.0f min — '
                           'remote kill switch (/halt, /flatten) is '
                           'unavailable: %s', (now - _tg_fail_since) / 60, e)
        return []


_tg_misconfig_warned = False


def notify(message: str, level: str = 'warning', dedupe_key: str | None = None):
    """Fire-and-forget alert. Safe to call from any trading path."""
    global _tg_misconfig_warned
    try:
        webhook = os.getenv('TRADER_WEBHOOK_URL')
        token = os.getenv('TRADER_TELEGRAM_BOT_TOKEN')
        chat = os.getenv('TRADER_TELEGRAM_CHAT_ID')
        if bool(token) != bool(chat) and not _tg_misconfig_warned:
            _tg_misconfig_warned = True
            logger.warning('[NOTIFY] telegram needs BOTH '
                           'TRADER_TELEGRAM_BOT_TOKEN and '
                           'TRADER_TELEGRAM_CHAT_ID — channel disabled')
        # _send() requires token AND chat: gating on the token alone would
        # record the dedupe entry and then provably send nothing
        if not (webhook or (token and chat)):
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
