# -*- coding: utf-8 -*-
"""
Adaptador al motor local (pruebas_robustez/motor/model_iacatching.py).
No depende de carpetas fuera de pruebas_robustez.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

SCRIPTS_DIR = Path(__file__).resolve().parent
PRUEBAS_DIR = SCRIPTS_DIR.parent
MOTOR_DIR = PRUEBAS_DIR / "motor"

if str(PRUEBAS_DIR) not in sys.path:
    sys.path.insert(0, str(PRUEBAS_DIR))

CHATBOT_KEYS = ("hayek", "hazlitt", "mises", "muso", "general")

CHATBOT_LABELS = {
    "hayek": "Friedrich A. Hayek",
    "hazlitt": "Henry Hazlitt",
    "mises": "Ludwig von Mises",
    "muso": "Manuel F. Ayau (Muso)",
    "general": "Todos los autores",
}

_RUNNERS: dict[str, Callable[..., Iterator[dict[str, Any]]]] | None = None
_MODEL_IDS: dict[str, str] | None = None


def ensure_pruebas_on_path() -> Path:
    if str(PRUEBAS_DIR) not in sys.path:
        sys.path.insert(0, str(PRUEBAS_DIR))
    return PRUEBAS_DIR


def load_runners() -> dict[str, Callable[..., Iterator[dict[str, Any]]]]:
    """Importa las chains del motor local (requiere red/AWS). Lazy para --dry-run."""
    global _RUNNERS, _MODEL_IDS
    if _RUNNERS is not None:
        return _RUNNERS

    ensure_pruebas_on_path()
    motor_file = MOTOR_DIR / "model_iacatching.py"
    if not motor_file.exists():
        raise FileNotFoundError(
            f"No se encontro el motor local: {motor_file}. "
            "Debe existir pruebas_robustez/motor/model_iacatching.py"
        )

    from motor import model_iacatching as mia  # noqa: WPS433

    _RUNNERS = {
        "hayek": mia.run_hayek_chain,
        "hazlitt": mia.run_hazlitt_chain,
        "mises": mia.run_mises_chain,
        "muso": mia.run_muso_chain,
        "general": mia.run_general_chain,
    }
    _MODEL_IDS = {
        "CHAT": getattr(mia, "model_id_chat", ""),
        "RENAME": getattr(mia, "model_id_rename", ""),
        "IS_TESTING": str(getattr(mia, "IS_TESTING", "")),
        "MOTOR": str(motor_file),
    }
    return _RUNNERS


def get_model_metadata() -> dict[str, str]:
    load_runners()
    return dict(_MODEL_IDS or {})


def ask(
    chatbot_key: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_retries: int = 10,
    retry_sleep_seconds: float = 2.0,
) -> tuple[str, float, str | None]:
    """
    Invoca la chain como la UI: history puede incluir ya el turno user actual;
    la chain vuelve a anadir el human final (mismo patron que pages/*).

    Returns:
        (respuesta_verbatim, latencia_segundos, error_o_None)
    """
    if chatbot_key not in CHATBOT_KEYS:
        raise ValueError(f"chatbot_key invalido: {chatbot_key!r}. Validos: {CHATBOT_KEYS}")

    runners = load_runners()
    run_fn = runners[chatbot_key]
    hist = list(history or [])

    last_error: str | None = None
    t0 = time.perf_counter()

    for attempt in range(1, max_retries + 1):
        try:
            full_response = ""
            for chunk in run_fn(question, hist):
                if isinstance(chunk, dict) and chunk.get("response"):
                    full_response += chunk["response"]
            latencia = time.perf_counter() - t0
            return full_response, latencia, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"intento {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)

    latencia = time.perf_counter() - t0
    return "", latencia, last_error
