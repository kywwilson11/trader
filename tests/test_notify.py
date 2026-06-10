"""Tests for notify.py kill-switch controls (halt/flatten flags, Telegram polling)."""




class TestKillSwitch:
    def test_halt_flag_roundtrip(self, tmp_path, monkeypatch):
        import notify
        monkeypatch.setattr(notify, '_HALT_FLAG', str(tmp_path / 'halt.flag'))
        assert not notify.halt_active()
        notify.set_halt('test reason')
        assert notify.halt_active()
        notify.clear_halt()
        assert not notify.halt_active()
        notify.clear_halt()  # idempotent on missing file

    def test_flatten_flag_roundtrip(self, tmp_path, monkeypatch):
        import notify
        monkeypatch.setattr(notify, '_FLATTEN_FLAG', str(tmp_path / 'f.flag'))
        assert not notify.flatten_requested()
        notify.request_flatten('tg')
        assert notify.flatten_requested()
        notify.clear_flatten_request()
        assert not notify.flatten_requested()

    def test_telegram_poll_filters_foreign_chats(self, tmp_path, monkeypatch):
        import io, json as _json, urllib.request
        import notify
        monkeypatch.setattr(notify, '_TG_OFFSET_FILE',
                            str(tmp_path / 'off.json'))
        monkeypatch.setenv('TRADER_TELEGRAM_BOT_TOKEN', 'tok')
        monkeypatch.setenv('TRADER_TELEGRAM_CHAT_ID', '42')
        payload = {'result': [
            {'update_id': 7, 'message': {'chat': {'id': 42},
                                         'text': '/halt'}},
            {'update_id': 8, 'message': {'chat': {'id': 666},
                                         'text': '/flatten'}},  # stranger
            {'update_id': 9, 'message': {'chat': {'id': 42},
                                         'text': '/STATUS@traderbot now'}},
        ]}
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: io.BytesIO(
                                _json.dumps(payload).encode()))
        cmds = notify.poll_telegram_commands()
        assert cmds == ['/halt', '/status']
        # Offset persisted -> a second poll with no new updates is empty
        payload['result'] = []
        assert notify.poll_telegram_commands() == []
        with open(tmp_path / 'off.json') as f:
            assert _json.load(f)['offset'] == 9

    def test_telegram_poll_never_raises(self, monkeypatch):
        import urllib.request
        import notify
        monkeypatch.setenv('TRADER_TELEGRAM_BOT_TOKEN', 'tok')
        monkeypatch.setenv('TRADER_TELEGRAM_CHAT_ID', '42')

        def boom(req, timeout=10):
            raise OSError('net down')

        monkeypatch.setattr(urllib.request, 'urlopen', boom)
        assert notify.poll_telegram_commands() == []

    def test_telegram_poll_disabled_without_env(self, monkeypatch):
        import notify
        monkeypatch.delenv('TRADER_TELEGRAM_BOT_TOKEN', raising=False)
        monkeypatch.delenv('TRADER_TELEGRAM_CHAT_ID', raising=False)
        assert notify.poll_telegram_commands() == []
