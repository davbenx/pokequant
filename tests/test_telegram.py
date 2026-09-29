"""
tests/test_telegram.py — Test per poke_quant/notify/telegram.py.

BUG TROVATO (audit performance/bug richiesto dall'utente, 2026-09-29): i
messaggi reali inviati (signal_scanner.py, slab_notifier.py) formattano il
testo con *grassetto* Markdown, ma send_telegram_message() non passava mai
parse_mode - l'utente vedeva gli asterischi letterali su Telegram invece del
testo in grassetto.
"""

from unittest.mock import patch, MagicMock

from poke_quant.notify.telegram import send_telegram_message


@patch("poke_quant.notify.telegram.requests.post")
def test_send_telegram_message_uses_markdown_parse_mode(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    ok = send_telegram_message("*grassetto* normale", token="t", chat_id="c")

    assert ok is True
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["parse_mode"] == "Markdown"
    assert kwargs["data"]["text"] == "*grassetto* normale"


def test_send_telegram_message_skips_silently_without_credentials():
    assert send_telegram_message("test", token=None, chat_id=None) is False
