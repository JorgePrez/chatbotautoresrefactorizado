# -*- coding: utf-8 -*-
"""Exportacion de resultados a CSV (plantilla) + metadatos JSON."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TZ_GT = ZoneInfo("America/Guatemala")

FIELDNAMES = [
    "id",
    "chatbot_utilizado",
    "fecha_hora_ejecucion",
    "respuesta_verbatim",
    "latencia_segundos",
    "observaciones",
]


def flatten_cell(value: Any) -> str:
    """Convierte saltos de linea/tabs a espacios para CSV legible en una sola linea."""
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def now_guatemala_iso() -> str:
    return datetime.now(TZ_GT).isoformat(timespec="seconds")


def load_existing_results(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    by_id: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            if rid:
                by_id[rid] = {k: (row.get(k) or "") for k in FIELDNAMES}
    return by_id


def write_results_csv(csv_path: Path, rows: list[dict[str, Any]], ordered_ids: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for rid in ordered_ids:
            row = by_id.get(rid) or {"id": rid}
            out = {k: flatten_cell(row.get(k, "")) for k in FIELDNAMES}
            out["id"] = rid
            writer.writerow(out)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "generado_en": now_guatemala_iso(),
        "timezone": "America/Guatemala",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_detalle_multi(path: Path, prueba_id: str, detalle: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"id": prueba_id, "turnos": detalle}, f, ensure_ascii=False, indent=2)


def is_completed_row(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    return bool((row.get("respuesta_verbatim") or "").strip())
