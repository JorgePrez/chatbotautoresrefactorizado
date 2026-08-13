# -*- coding: utf-8 -*-
"""Reextrae referencia_prompts_sistema desde motor/model_iacatching.py local."""
from __future__ import annotations

import json
import re
from pathlib import Path

PRUEBAS = Path(__file__).resolve().parents[1]
MOTOR = PRUEBAS / "motor" / "model_iacatching.py"
OUT = PRUEBAS / "referencia_prompts_sistema"


def main() -> None:
    src = MOTOR.read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    index = {"origen": "motor/model_iacatching.py", "archivos": {}}
    for name in ["HAYEK", "HAZLITT", "MISES", "GENERAL", "MUSO"]:
        m = re.search(rf'SYSTEM_PROMPT_{name}\s*=\s*"""(.*?)"""', src, flags=re.S)
        if not m:
            raise SystemExit(f"missing SYSTEM_PROMPT_{name}")
        body = m.group(1).strip() + "\n"
        fname = f"system_prompt_{name.lower()}.txt"
        (OUT / fname).write_text(body, encoding="utf-8", newline="\n")
        index["archivos"][name.lower()] = {
            "archivo": fname,
            "palabras": len(body.split()),
            "chars": len(body),
        }
        print("wrote", fname, len(body.split()))
    (OUT / "indice.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("ok")


if __name__ == "__main__":
    main()
