# -*- coding: utf-8 -*-
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "ANALISIS_Y_PLAN_AUTOMATIZACION.md"

TEXT = r'''# Analisis y plan - Automatizacion de la bateria de robustez CHH

**Fecha:** 12 de agosto de 2026 (actualizado)  
**Estado:** listo para ejecutar. Directorio **autocontenido**.  
**Detalle operativo:** ver `INSTRUCCIONES_EJECUCION.md`.

---

## 1. Objetivo

Ejecutar las **35 pruebas** de la bateria de robustez CHH de forma automatica (sin UI Streamlit ni DynamoDB), capturando respuestas verbatim + latencia + metadatos.

No puntua aprobado/fallido: solo ejecuta y registra.

---

## 2. Independencia del directorio

`pruebas_robustez` puede abrirse y correrse **solo**, sin depender de `config/` ni del resto del repo.

| Necesita | No necesita |
|---|---|
| Esta carpeta completa | `config/` del repo padre |
| Anaconda base (langchain-aws, boto3, etc.) | Streamlit / login Google |
| Credenciales AWS + red (Bedrock/KB) | DynamoDB |

Motor local: `motor/model_iacatching.py` (copia del motor de produccion).

---

## 3. Estructura preparada

```
pruebas_robustez/
  motor/
    model_iacatching.py      # system prompts + chains + Knowledge Bases
    README.md
  scripts/
    chatbot_client.py        # adaptador al motor local
    runner_robustez.py       # orquestador CLI (las 35)
    multi_turno.py           # R34 / R35
    export_resultados.py     # CSV + meta.json
    smoke_check.py           # 1 pregunta por autor
  fixtures/
    r32_apuntes.txt          # pegado largo R32 (~3000 palabras)
    r35_secuencia.json       # 12 previas + pregunta memoria
  referencia_prompts_sistema/
    system_prompt_*.txt      # copia legible (no altera ejecucion)
  salidas/                   # resultados CSV/JSON
  bateria_robustez_chh_v1.json
  bateria_robustez_chh_v1.csv
  plantilla_resultados_chh.csv
  PROTOCOLO_EJECUCION.md
  INSTRUCCIONES_EJECUCION.md
  ANALISIS_Y_PLAN_AUTOMATIZACION.md
  .vscode/settings.json      # UTF-8 (workspace de esta carpeta)
```

---

## 4. Que se carga en cada prueba

### Prompts de usuario (la bateria)
- Fuente: `bateria_robustez_chh_v1.json`
- R32: `fixtures/r32_apuntes.txt` + "Completalo y corrigelo"
- R35: `fixtures/r35_secuencia.json` (multi-turno)

### System prompts + modelo + KB
- Fuente viva: `motor/model_iacatching.py`
- `IS_TESTING = False` (produccion tal cual)
- Knowledge Bases definidas ahi:

| chatbot_key | KB ID |
|---|---|
| hayek | HME7HA8YXX |
| hazlitt | 7MFCUWJSJJ |
| mises | 4L0WE8NOOH |
| general | WGUUTHDVPH |
| muso | HE8WRDDBFH |

---

## 5. Decisiones confirmadas

| # | Tema | Decision |
|---|---|---|
| 1 | Entorno | Produccion tal cual (`IS_TESTING=False`) |
| 2 | `cualquiera` | Round-robin: hayek, hazlitt, mises, muso, general |
| 3 | Persistencia | Solo archivos en `salidas/` (sin DynamoDB) |
| 4 | R32 | Fixture local de apuntes de una clase (~3000+ palabras) |
| 5 | R35 | Secuencia fija de 12 + pregunta final |
| 6 | Entrega | Scripts listos; el usuario lanza la corrida |
| 7 | UTF-8 | Workspace de esta carpeta |
| 8 | Autonomia | Motor vendorizado en `motor/` |

Round-robin concreto:
R01->hayek, R02->hazlitt, R03->mises, R06->muso, R07->general, R08->hayek

---

## 6. Distribucion de pruebas

| Cat. | Nombre | N |
|---|---|---|
| A | Fuera de dominio | 5 |
| B | Manipulacion | 5 |
| C | Integridad academica | 3 |
| D | Abstencion | 5 |
| E | Cruce de autores | 3 |
| F | Persona sensible | 4 |
| G | Presion del usuario | 4 |
| H | Formato | 4 |
| I | Multi-turno | 2 |

Prioridad: ALTA 20, MEDIA 9, BAJA 6.

---

## 7. Flujo de ejecucion

1. Smoke check (1 pregunta por autor)
2. Dry-run (sin Bedrock; valida plan)
3. Corrida real de las 35
4. Revisar `salidas/*.csv` + `*.meta.json`
5. Entregar resultados (sin puntuar)

Tiempo orientativo de la corrida completa: 30-90 min.

---

## 8. Criterio de exito

- Cada ID con `respuesta_verbatim` (o error tecnico en observaciones)
- `chatbot_utilizado` y `fecha_hora_ejecucion` rellenados
- Metadatos en `.meta.json`
- Reproducible con Anaconda + `--resume`

---

## 9. Comandos (Anaconda)

```powershell
cd <ruta>\pruebas_robustez\scripts

C:\Users\IT-14\anaconda3\python.exe .\smoke_check.py
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --dry-run
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py
```

Opciones utiles:

```powershell
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --solo-alta
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --ids R01,R14,R22
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --resume --salida ..\salidas\resultados_chh_XXXX.csv
```

---

## 10. Notas

- Editar `referencia_prompts_sistema/*.txt` no cambia el comportamiento.
- Para alinear el motor local con produccion (si el repo padre esta disponible):
  `copy ..\config\model_iacatching.py .\motor\model_iacatching.py`
- No se automatiza la UI Streamlit.
'''


def main() -> None:
    OUT.write_text(TEXT.lstrip("\n"), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} chars={len(TEXT)}")


if __name__ == "__main__":
    main()
