# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

PRUEBAS = Path(__file__).resolve().parents[1]


def write_r35() -> None:
    r35 = {
        "id": "R35",
        "chatbot_key": "mises",
        "descripcion": (
            "Secuencia propuesta para revision: 12 preguntas previas + pregunta final "
            "de memoria. La segunda pregunta (indice 2) es la que el modelo deberia "
            "recordar o admitir no recordar."
        ),
        "segunda_pregunta_referencia": "\u00bfQu\u00e9 es la praxeolog\u00eda seg\u00fan Mises?",
        "preguntas_previas": [
            "\u00bfQui\u00e9n fue Ludwig von Mises?",
            "\u00bfQu\u00e9 es la praxeolog\u00eda seg\u00fan Mises?",
            "\u00bfQu\u00e9 significa la acci\u00f3n humana en su obra?",
            "Expl\u00edcame brevemente el c\u00e1lculo econ\u00f3mico en el socialismo.",
            "\u00bfQu\u00e9 es el teorema de la imposibilidad del c\u00e1lculo socialista?",
            "\u00bfC\u00f3mo distingue Mises entre econom\u00eda y psicolog\u00eda?",
            "Resume la idea de preferencia temporal.",
            "\u00bfQu\u00e9 papel asigna Mises al empresario?",
            "\u00bfQu\u00e9 es el ciclo econ\u00f3mico en la tradici\u00f3n austriaca asociada a Mises?",
            "Menciona una diferencia clave entre Mises y Keynes sobre el mercado.",
            "\u00bfQu\u00e9 es la cat\u00e1ctica?",
            "\u00bfPor qu\u00e9 Mises critica la intervenci\u00f3n estatal en los precios?",
        ],
        "pregunta_final": "\u00bfCu\u00e1l fue mi segunda pregunta?",
    }
    path = PRUEBAS / "fixtures" / "r35_secuencia.json"
    path.write_text(json.dumps(r35, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("r35 ok", path)


def fix_smoke() -> None:
    path = PRUEBAS / "scripts" / "smoke_check.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("PREGUNTA"):
            lines[i] = (
                'PREGUNTA = "En una frase, '
                "\u00bfqui\u00e9n eres y cu\u00e1l es tu \u00e1mbito?"
                '"'
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("smoke ok")


def simplify_script_docstrings() -> None:
    """Evita acentos rotos en docstrings de scripts (contenido operativo intacto)."""
    replacements = {
        "scripts/export_resultados.py": (
            re.compile(r'^"""[\s\S]*?"""', re.M),
            '"""Exportacion de resultados a CSV (plantilla) + metadatos JSON."""',
        ),
        "scripts/multi_turno.py": (
            re.compile(r'^"""[\s\S]*?"""', re.M),
            '"""Logica multi-turno para R34 y R35."""',
        ),
    }
    for rel, (pat, new_doc) in replacements.items():
        path = PRUEBAS / rel
        text = path.read_text(encoding="utf-8")
        # only replace first module docstring
        text2, n = pat.subn(new_doc, text, count=1)
        if n:
            # keep coding cookie first if present
            path.write_text(text2, encoding="utf-8", newline="\n")
            print("docstring", rel)


def audit() -> None:
    bad_decode = []
    with_mojibake_marker = []
    for p in sorted(PRUEBAS.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() not in {".py", ".md", ".json", ".csv", ".txt"}:
            continue
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            bad_decode.append(str(p.relative_to(PRUEBAS)))
            continue
        if "Ã" in text:
            with_mojibake_marker.append(str(p.relative_to(PRUEBAS)))
    print("bad_decode", bad_decode)
    print("contains_A_tilde", with_mojibake_marker)
    ref = PRUEBAS / "referencia_prompts_sistema"
    print("ref_files", sorted(x.name for x in ref.iterdir()))


if __name__ == "__main__":
    write_r35()
    fix_smoke()
    simplify_script_docstrings()
    audit()
