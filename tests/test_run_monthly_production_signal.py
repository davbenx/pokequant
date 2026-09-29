"""
tests/test_run_monthly_production_signal.py — Test per scripts/run_monthly_production_signal.py.

BUG TROVATO (audit performance/bug richiesto dall'utente, 2026-09-29):
run_step() non segnalava mai un fallimento oltre a un print su stdout -
main() proseguiva comunque attraverso tutti gli step successivi (incluso
l'invio del segnale via Telegram) indipendentemente dall'esito di
rebuild_prices_with_real_fx.py/flag_unreliable_assets.py, presentando dati
potenzialmente vecchi come se fossero freschi, senza nessuna traccia visibile
all'utente (solo nel log stdout della GitHub Action, che nessuno legge di
routine).
"""

from unittest.mock import patch, MagicMock

from scripts.run_monthly_production_signal import run_step, _prepend_failure_warning


@patch("scripts.run_monthly_production_signal.subprocess.run")
def test_run_step_returns_true_on_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert run_step("ok step", ["scripts/fake.py"]) is True


@patch("scripts.run_monthly_production_signal.subprocess.run")
def test_run_step_returns_false_on_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1)
    assert run_step("failing step", ["scripts/fake.py"]) is False


def test_prepend_failure_warning_noop_when_no_failures():
    assert _prepend_failure_warning("segnale mensile", []) == "segnale mensile"


def test_prepend_failure_warning_flags_failed_steps():
    msg = _prepend_failure_warning("segnale mensile", ["Ricostruzione prezzi con FX reale"])
    assert msg.startswith("🚨 ATTENZIONE")
    assert "Ricostruzione prezzi con FX reale" in msg
    assert msg.endswith("segnale mensile")
