# -*- coding: utf-8 -*-
"""Valida fixtures/r32_apuntes.txt (no lo sobrescribe)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "r32_apuntes.txt"


def main() -> None:
    text = OUT.read_text(encoding="utf-8")
    words = len(text.split())
    print(f"words={words} chars={len(text)} -> {OUT}")
    if words < 3000:
        raise SystemExit("corto (<3000 palabras)")
    if words > 4500:
        raise SystemExit("largo (>4500)")
    low = text.lower()
    for bad in ("verificar", "revision rapida", "ensayo", "tutoria", "fixture", "r32"):
        if bad in low:
            raise SystemExit(f"indeseado: {bad}")
    if text.count("26. Relacion entre los bloques") != 1:
        raise SystemExit("posible repeticion o falta seccion 26")
    print("ok")


if __name__ == "__main__":
    main()
