# -*- coding: utf-8 -*-
"""
Orquestador de la bateria de robustez CHH.

Uso (Anaconda base):
  C:\\Users\\IT-14\\anaconda3\\python.exe runner_robustez.py --dry-run
  C:\\Users\\IT-14\\anaconda3\\python.exe runner_robustez.py --solo-alta
  C:\\Users\\IT-14\\anaconda3\\python.exe runner_robustez.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parent
PRUEBAS_DIR = SCRIPTS_DIR.parent
FIXTURES_DIR = PRUEBAS_DIR / "fixtures"
SALIDAS_DIR = PRUEBAS_DIR / "salidas"
TZ_GT = ZoneInfo("America/Guatemala")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatbot_client import (  # noqa: E402
    CHATBOT_KEYS,
    CHATBOT_LABELS,
    ask,
    get_model_metadata,
)
from export_resultados import (  # noqa: E402
    is_completed_row,
    load_existing_results,
    now_guatemala_iso,
    write_detalle_multi,
    write_metadata,
    write_results_csv,
)
from multi_turno import run_r34, run_r35  # noqa: E402

BATERIA_JSON = PRUEBAS_DIR / "bateria_robustez_chh_v1.json"
R32_FIXTURE = FIXTURES_DIR / "r32_apuntes.txt"
R35_FIXTURE = FIXTURES_DIR / "r35_secuencia.json"
ROUND_ROBIN_ORDER = list(CHATBOT_KEYS)  # hayek, hazlitt, mises, muso, general


def now_iso() -> str:
    """Fecha/hora de ejecucion en zona America/Guatemala."""
    return now_guatemala_iso()


def load_bateria(path: Path = BATERIA_JSON) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def assign_cualquiera(pruebas: list[dict[str, Any]]) -> dict[str, str]:
    """Round-robin estable por orden de aparicion de IDs con chatbot_key=cualquiera."""
    mapping: dict[str, str] = {}
    i = 0
    for p in pruebas:
        if p.get("chatbot_key") == "cualquiera":
            mapping[p["id"]] = ROUND_ROBIN_ORDER[i % len(ROUND_ROBIN_ORDER)]
            i += 1
    return mapping


def resolve_chatbot_key(prueba: dict[str, Any], cualquiera_map: dict[str, str]) -> str:
    key = prueba.get("chatbot_key") or "hayek"
    if key == "cualquiera":
        return cualquiera_map[prueba["id"]]
    if key not in CHATBOT_KEYS:
        raise ValueError(f"{prueba['id']}: chatbot_key desconocido {key!r}")
    return key


# Bedrock Retrieve: retrievalQuery.text <= 20000 caracteres.
_R32_RETRIEVE_CHAR_LIMIT = 20000
_R32_SUFFIX = "\n\nCompl\u00e9talo y corr\u00edgelo"


def build_r32_prompt() -> str:
    """Arma el prompt R32: contenido de fixtures/r32_apuntes.txt + instruccion final."""
    if not R32_FIXTURE.exists():
        raise FileNotFoundError(
            f"Falta fixture R32: {R32_FIXTURE}. Reviselo antes de la corrida completa."
        )
    texto = R32_FIXTURE.read_text(encoding="utf-8").strip()
    # Compactar blancos: Retrieve falla si retrievalQuery.text > 20000.
    texto = "\n".join(ln.rstrip() for ln in texto.splitlines() if ln.strip())
    palabras = len(texto.split())
    if palabras < 2500:
        raise ValueError(
            f"R32 fixture tiene ~{palabras} palabras; el protocolo pide ~3000+. "
            f"Amplie {R32_FIXTURE}"
        )
    budget = _R32_RETRIEVE_CHAR_LIMIT - len(_R32_SUFFIX)
    if len(texto) > budget:
        cut = texto[:budget]
        sp = cut.rfind(" ")
        if sp > budget - 200:
            cut = cut[:sp]
        texto = cut.rstrip()
        palabras = len(texto.split())
        if palabras < 2500:
            raise ValueError(
                f"R32 tras recorte por limite Retrieve ({_R32_RETRIEVE_CHAR_LIMIT} chars) "
                f"quedo en ~{palabras} palabras; reduzca blancos o densifique "
                f"{R32_FIXTURE}"
            )
    prompt = f"{texto}{_R32_SUFFIX}"
    if len(prompt) > _R32_RETRIEVE_CHAR_LIMIT:
        raise ValueError(
            f"R32 prompt tiene {len(prompt)} chars; Retrieve admite max "
            f"{_R32_RETRIEVE_CHAR_LIMIT}."
        )
    return prompt


def filter_pruebas(
    pruebas: list[dict[str, Any]],
    *,
    ids: set[str] | None,
    solo_alta: bool,
) -> list[dict[str, Any]]:
    out = []
    for p in pruebas:
        if ids and p["id"] not in ids:
            continue
        if solo_alta and (p.get("prioridad") or "").upper() != "ALTA":
            continue
        out.append(p)
    return out


def run_one(
    prueba: dict[str, Any],
    chatbot_key: str,
    *,
    dry_run: bool,
    max_retries: int,
) -> dict[str, Any]:
    pid = prueba["id"]
    label = CHATBOT_LABELS[chatbot_key]
    tipo = (prueba.get("tipo_sesion") or "").lower()
    observaciones_parts: list[str] = []

    if prueba.get("chatbot_key") == "cualquiera":
        observaciones_parts.append(f"cualquiera->{chatbot_key}")

    if dry_run:
        preview = (prueba.get("prompt") or "")[:80].replace("\n", " ")
        print(f"[dry-run] {pid} -> {chatbot_key} ({label}) | {tipo} | {preview}...")
        if pid == "R32":
            observaciones_parts.append("depende de fixtures/r32_apuntes.txt")
        if pid == "R34":
            observaciones_parts.append("multi-turno; se genera salidas/detalle_R34.json")
        if pid == "R35":
            observaciones_parts.append("depende de fixtures/r35_secuencia.json")
            observaciones_parts.append("multi-turno; se genera salidas/detalle_R35.json")
        return {
            "id": pid,
            "chatbot_utilizado": label,
            "fecha_hora_ejecucion": "",
            "respuesta_verbatim": "",
            "latencia_segundos": "",
            "observaciones": "; ".join(observaciones_parts + ["dry-run"]),
        }

    print(f">>> {pid} -> {chatbot_key} ({label})")

    if pid == "R34" or "multi-turno" in tipo and pid == "R34":
        resp, lat, detalle, err = run_r34(chatbot_key, prueba["prompt"], max_retries=max_retries)
        detalle_path = SALIDAS_DIR / f"detalle_{pid}.json"
        write_detalle_multi(detalle_path, pid, detalle)
        observaciones_parts.append("multi-turno; existe salidas/detalle_R34.json")
        if err:
            observaciones_parts.append(err)
        return {
            "id": pid,
            "chatbot_utilizado": label,
            "fecha_hora_ejecucion": now_iso(),
            "respuesta_verbatim": resp,
            "latencia_segundos": f"{lat:.3f}",
            "observaciones": "; ".join(observaciones_parts),
        }

    if pid == "R35" or ("multi-turno" in tipo and pid == "R35"):
        resp, lat, detalle, err = run_r35(chatbot_key, max_retries=max_retries)
        detalle_path = SALIDAS_DIR / f"detalle_{pid}.json"
        write_detalle_multi(detalle_path, pid, detalle)
        observaciones_parts.append("depende de fixtures/r35_secuencia.json")
        observaciones_parts.append("multi-turno; existe salidas/detalle_R35.json")
        if err:
            observaciones_parts.append(err)
        return {
            "id": pid,
            "chatbot_utilizado": label,
            "fecha_hora_ejecucion": now_iso(),
            "respuesta_verbatim": resp,
            "latencia_segundos": f"{lat:.3f}",
            "observaciones": "; ".join(observaciones_parts),
        }

    if pid == "R32":
        prompt = build_r32_prompt()
        observaciones_parts.append("depende de fixtures/r32_apuntes.txt")
    else:
        prompt = prueba["prompt"]

    history = [{"role": "user", "content": prompt}]
    resp, lat, err = ask(chatbot_key, prompt, history, max_retries=max_retries)
    if err:
        observaciones_parts.append(err)

    return {
        "id": pid,
        "chatbot_utilizado": label,
        "fecha_hora_ejecucion": now_iso(),
        "respuesta_verbatim": resp,
        "latencia_segundos": f"{lat:.3f}",
        "observaciones": "; ".join(observaciones_parts),
    }


def default_salida_path() -> Path:
    stamp = datetime.now(TZ_GT).strftime("%Y%m%d_%H%M%S")
    return SALIDAS_DIR / f"resultados_chh_{stamp}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner bateria robustez CHH")
    parser.add_argument("--dry-run", action="store_true", help="No llama a Bedrock")
    parser.add_argument("--solo-alta", action="store_true", help="Solo prioridad ALTA")
    parser.add_argument("--ids", default="", help="IDs separados por coma, ej. R01,R14")
    parser.add_argument("--resume", action="store_true", help="Salta IDs ya con respuesta en --salida")
    parser.add_argument("--salida", type=Path, default=None, help="CSV de salida")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument(
        "--bateria",
        type=Path,
        default=BATERIA_JSON,
        help="JSON de la bateria",
    )
    args = parser.parse_args()

    bateria = load_bateria(args.bateria)
    pruebas_all: list[dict[str, Any]] = bateria["pruebas"]
    ordered_ids = [p["id"] for p in pruebas_all]
    cualquiera_map = assign_cualquiera(pruebas_all)

    ids = {x.strip() for x in args.ids.split(",") if x.strip()} or None
    pruebas = filter_pruebas(pruebas_all, ids=ids, solo_alta=args.solo_alta)

    salida = args.salida or default_salida_path()
    existing = load_existing_results(salida) if args.resume and salida.exists() else {}

    print(f"Pruebas dir: {PRUEBAS_DIR}")
    print(f"Pruebas a ejecutar: {len(pruebas)} (de {len(pruebas_all)})")
    print(f"Round-robin cualquiera: {cualquiera_map}")
    print(f"Salida: {salida}")
    if args.dry_run:
        print("Modo DRY-RUN (sin Bedrock)\n")
    else:
        print("Cargando motor local (motor/model_iacatching.py)...")
        meta = get_model_metadata()
        print(f"Modelo: {meta}\n")

    results_by_id: dict[str, dict[str, Any]] = dict(existing)
    errores = 0

    for prueba in pruebas:
        pid = prueba["id"]
        if args.resume and is_completed_row(existing.get(pid)):
            print(f"[resume] salto {pid} (ya tiene respuesta)")
            continue

        key = resolve_chatbot_key(prueba, cualquiera_map)
        try:
            row = run_one(prueba, key, dry_run=args.dry_run, max_retries=args.max_retries)
        except Exception as exc:  # noqa: BLE001
            errores += 1
            row = {
                "id": pid,
                "chatbot_utilizado": CHATBOT_LABELS.get(key, key),
                "fecha_hora_ejecucion": now_iso(),
                "respuesta_verbatim": "",
                "latencia_segundos": "",
                "observaciones": f"excepcion: {type(exc).__name__}: {exc}",
            }
            print(f"ERROR {pid}: {exc}")

        results_by_id[pid] = row
        if (row.get("observaciones") or "") and "fallo" in (row.get("observaciones") or "").lower():
            errores += 1
        if not args.dry_run and not (row.get("respuesta_verbatim") or "").strip():
            errores += 1

        # Persistencia incremental (sin DynamoDB)
        rows = [results_by_id.get(i, {"id": i}) for i in ordered_ids]
        # Rellenar chatbot precargado de plantilla si falta
        for p in pruebas_all:
            rid = p["id"]
            r = results_by_id.get(rid)
            if r and not r.get("chatbot_utilizado") and p.get("chatbot_key") != "cualquiera":
                r["chatbot_utilizado"] = CHATBOT_LABELS.get(p["chatbot_key"], p.get("chatbot", ""))
        write_results_csv(salida, list(results_by_id.values()), ordered_ids)

    meta_path = salida.with_suffix(".meta.json")
    write_metadata(
        meta_path,
        {
            "entorno": "produccion_tal_cual",
            "IS_TESTING": False,
            "persistencia": "solo_archivos_locales",
            "dynamodb": False,
            "cualquiera_round_robin": cualquiera_map,
            "python": sys.executable,
            "bateria": str(args.bateria),
            "salida_csv": str(salida),
            "dry_run": args.dry_run,
            "n_ejecutadas": len(pruebas),
            "modelo": ({} if args.dry_run else get_model_metadata()),
        },
    )

    print(f"\nCSV: {salida}")
    print(f"Meta: {meta_path}")
    if errores and not args.dry_run:
        print(f"Finalizado con {errores} incidencia(s). Revise observaciones.")
        return 1
    print("Finalizado OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
