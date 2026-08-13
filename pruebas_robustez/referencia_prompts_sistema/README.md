# Referencia de prompts de sistema (solo copia legible)

## Donde vive el motor real de la bateria

La ejecucion usa:

`pruebas_robustez/motor/model_iacatching.py`

Ahi estan embebidos:

- `SYSTEM_PROMPT_HAYEK`
- `SYSTEM_PROMPT_HAZLITT`
- `SYSTEM_PROMPT_MISES`
- `SYSTEM_PROMPT_GENERAL`
- `SYSTEM_PROMPT_MUSO`

y las funciones `run_*_chain`.

Esta carpeta `referencia_prompts_sistema/` solo facilita leer esos prompts en `.txt`.
Editar estos `.txt` **no** cambia el comportamiento.

## Prompts de las pruebas (usuario)

Vienen de:

- `bateria_robustez_chh_v1.json`
- `fixtures/r32_apuntes.txt` (R32)
- `fixtures/r35_secuencia.json` (R35)

## Independencia

`pruebas_robustez` ya no importa `config/` del repo padre.
Puede abrirse y ejecutarse solo (con Anaconda + AWS).
