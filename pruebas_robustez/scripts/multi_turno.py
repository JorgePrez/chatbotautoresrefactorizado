# -*- coding: utf-8 -*-
"""Logica multi-turno para R34 y R35."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chatbot_client import ask

PRUEBAS_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PRUEBAS_DIR / "fixtures"

# Turno 1: "..." / Turno 2: "..." / Turno 3: "..."
_TURNO_RE = re.compile(
    r'Turno\s+(\d+)\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


def parse_r34_turnos(prompt_field: str) -> list[str]:
    matches = _TURNO_RE.findall(prompt_field)
    if not matches:
        raise ValueError(
            "No se pudieron parsear turnos R34. Formato esperado: "
            'Turno 1: "..." / Turno 2: "..." / Turno 3: "..."'
        )
    matches_sorted = sorted(matches, key=lambda x: int(x[0]))
    return [texto for _, texto in matches_sorted]


def load_r35_secuencia(path: Path | None = None) -> dict[str, Any]:
    secuencia_path = path or (FIXTURES_DIR / "r35_secuencia.json")
    with secuencia_path.open(encoding="utf-8") as f:
        data = json.load(f)
    previas = data.get("preguntas_previas") or []
    final = data.get("pregunta_final") or ""
    if len(previas) < 10:
        raise ValueError(f"R35 requiere >=10 preguntas previas; hay {len(previas)} en {secuencia_path}")
    if not final:
        raise ValueError(f"R35 sin pregunta_final en {secuencia_path}")
    return data


def run_multi_turno(
    chatbot_key: str,
    turnos: list[str],
    *,
    max_retries: int = 10,
) -> tuple[str, float, list[dict[str, Any]], str | None]:
    """
    Ejecuta turnos en la misma conversación.
    Devuelve: (respuesta_ultimo_turno, latencia_total, detalle_por_turno, error)
    """
    history: list[dict[str, str]] = []
    detalle: list[dict[str, Any]] = []
    latencia_total = 0.0
    last_response = ""

    for i, pregunta in enumerate(turnos, start=1):
        history.append({"role": "user", "content": pregunta})
        respuesta, latencia, error = ask(
            chatbot_key,
            pregunta,
            history,
            max_retries=max_retries,
        )
        latencia_total += latencia
        detalle.append(
            {
                "turno": i,
                "pregunta": pregunta,
                "respuesta": respuesta,
                "latencia_segundos": round(latencia, 3),
                "error": error,
            }
        )
        if error:
            return last_response, latencia_total, detalle, f"fallo en turno {i}: {error}"
        history.append({"role": "assistant", "content": respuesta})
        last_response = respuesta

    return last_response, latencia_total, detalle, None


def run_r34(chatbot_key: str, prompt_field: str, **kwargs: Any) -> tuple[str, float, list[dict[str, Any]], str | None]:
    turnos = parse_r34_turnos(prompt_field)
    return run_multi_turno(chatbot_key, turnos, **kwargs)


def run_r35(chatbot_key: str, secuencia_path: Path | None = None, **kwargs: Any) -> tuple[str, float, list[dict[str, Any]], str | None]:
    data = load_r35_secuencia(secuencia_path)
    turnos = list(data["preguntas_previas"]) + [data["pregunta_final"]]
    return run_multi_turno(chatbot_key, turnos, **kwargs)
