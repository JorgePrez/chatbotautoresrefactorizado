# -*- coding: utf-8 -*-
"""Revisa UTF-8 en pruebas_robustez y copia system prompts de referencia."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRUEBAS = Path(__file__).resolve().parents[1]
MODEL = ROOT / "config" / "model_iacatching.py"
OUT_DIR = PRUEBAS / "referencia_prompts_sistema"


def fix_mojibake_in_pruebas() -> list[str]:
    fixed: list[str] = []
    for p in PRUEBAS.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() not in {".py", ".md", ".json", ".csv", ".txt"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "Ãƒ" not in text and "Ã¢" not in text:
            continue
        repaired = None
        try:
            candidate = text.encode("cp1252").decode("utf-8")
            if "Ãƒ" not in candidate:
                repaired = candidate
        except Exception:
            repaired = None
        if repaired is None:
            repaired = (
                text.replace("batería", "batería")
                .replace("Revíselo", "Revíselo")
                .replace("Amplíe", "Amplíe")
                .replace("Complétalo y corrígelo", "Complétalo y corrígelo")
            )
        if repaired != text:
            p.write_text(repaired, encoding="utf-8", newline="\n")
            fixed.append(str(p.relative_to(PRUEBAS)))
    return fixed


def extract_system_prompts() -> dict:
    src = MODEL.read_text(encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)
    names = ["HAYEK", "HAZLITT", "MISES", "GENERAL", "MUSO"]
    index = {
        "origen": "config/model_iacatching.py (constantes SYSTEM_PROMPT_*)",
        "nota": (
            "Copia de referencia UTF-8 para inspeccionar. "
            "La corrida real NO lee estos archivos: importa model_iacatching."
        ),
        "archivos": {},
    }
    for name in names:
        pat = rf'SYSTEM_PROMPT_{name}\s*=\s*"""(.*?)"""'
        m = re.search(pat, src, flags=re.S)
        if not m:
            raise RuntimeError(f"No se encontro SYSTEM_PROMPT_{name}")
        body = m.group(1).strip() + "\n"
        fname = f"system_prompt_{name.lower()}.txt"
        (OUT_DIR / fname).write_text(body, encoding="utf-8", newline="\n")
        index["archivos"][name.lower()] = {
            "archivo": fname,
            "chars": len(body),
            "palabras": len(body.split()),
            "chatbot_key": "general" if name == "GENERAL" else name.lower(),
        }
        print(f"wrote {fname} words={len(body.split())}")
    return index


def write_readme() -> None:
    readme = """# Referencia de prompts de sistema (solo copia)

## Como funciona en la app real

Los **system prompts** (identidad de cada autor) **no** se cargan desde archivos `.txt` en `config/`.
Estan embebidos como constantes en:

`config/model_iacatching.py`

- `SYSTEM_PROMPT_HAYEK`
- `SYSTEM_PROMPT_HAZLITT`
- `SYSTEM_PROMPT_MISES`
- `SYSTEM_PROMPT_GENERAL` (Todos los autores)
- `SYSTEM_PROMPT_MUSO`

Las pages (`pages/hayek.py`, etc.) importan `run_*_chain` desde `model_iacatching`.
Cada `run_*_chain` inyecta el system prompt correspondiente + contexto RAG + historial + pregunta del usuario.

## Que usa la bateria de robustez

1. **Prompts de prueba (usuario):** `pruebas_robustez/bateria_robustez_chh_v1.json`
   - Excepciones: R32 lee `fixtures/r32_apuntes.txt`; R35 lee `fixtures/r35_secuencia.json`
2. **System prompts + modelo + KB:** se usan **en vivo** importando `config.model_iacatching`
   (igual que produccion). Esta carpeta es solo para leer/inspeccionar.

## Importante

- Editar estos `.txt` **no** cambia el comportamiento del runner.
- Para cambiar el comportamiento real habria que tocar `config/model_iacatching.py`.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def audit_utf8() -> None:
    bad = []
    for p in sorted(PRUEBAS.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in {".pyc", ".png", ".jpg"}:
            continue
        raw = p.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            bad.append(str(p.relative_to(PRUEBAS)))
    if bad:
        raise SystemExit(f"No UTF-8: {bad}")
    print("utf8 audit: OK")


def main() -> None:
    print("fixed:", fix_mojibake_in_pruebas())
    index = extract_system_prompts()
    write_readme()
    (OUT_DIR / "indice.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_utf8()
    bateria = json.loads((PRUEBAS / "bateria_robustez_chh_v1.json").read_text(encoding="utf-8"))
    print("R01 sample:", repr(bateria["pruebas"][0]["prompt"][:50]))


if __name__ == "__main__":
    main()
