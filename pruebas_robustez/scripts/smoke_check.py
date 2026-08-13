# -*- coding: utf-8 -*-
"""
Smoke check: una pregunta inocua por autor para validar AWS + Knowledge Bases.
No usa DynamoDB ni Streamlit UI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatbot_client import CHATBOT_KEYS, CHATBOT_LABELS, ask, get_model_metadata  # noqa: E402

PREGUNTA = "En una frase, ¿quién eres y cuál es tu ámbito?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke check de las 5 chains CHH")
    parser.add_argument(
        "--autores",
        default=",".join(CHATBOT_KEYS),
        help="Lista separada por comas (default: los 5)",
    )
    args = parser.parse_args()
    autores = [a.strip() for a in args.autores.split(",") if a.strip()]

    print("Cargando motor local motor/model_iacatching.py (red/AWS)...")
    meta = get_model_metadata()
    print(f"Metadatos modelo: {meta}")
    print(f"Pregunta smoke: {PREGUNTA!r}\n")

    fallos = 0
    for key in autores:
        if key not in CHATBOT_KEYS:
            print(f"[SKIP] {key}: key desconocido")
            fallos += 1
            continue
        label = CHATBOT_LABELS[key]
        print(f"=== {key} ({label}) ===")
        history = [{"role": "user", "content": PREGUNTA}]
        resp, lat, err = ask(key, PREGUNTA, history, max_retries=3)
        if err:
            fallos += 1
            print(f"ERROR ({lat:.1f}s): {err}\n")
            continue
        preview = (resp or "").replace("\n", " ")[:240]
        print(f"OK ({lat:.1f}s): {preview}...\n" if len(resp) > 240 else f"OK ({lat:.1f}s): {preview}\n")

    if fallos:
        print(f"Smoke FINALIZADO con {fallos} fallo(s).")
        return 1
    print("Smoke OK en todos los autores solicitados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
