# Instrucciones de ejecucion - Bateria de robustez CHH (automatizada)

**Interprete obligatorio:** Anaconda base  
`C:\Users\IT-14\anaconda3\python.exe`

**Independencia:** este directorio `pruebas_robustez` es **autocontenido**.
No necesita la carpeta `config/` ni el resto del repo para ejecutarse.
El motor vive en `motor/model_iacatching.py` (copia local).

**Decisiones fijadas:** produccion tal cual · round-robin en `cualquiera` · sin DynamoDB · fixtures R32/R35 · UTF-8 en esta carpeta.

**Dependencias externas (no son carpetas del repo):**
- Paquetes Python en Anaconda: `boto3`, `langchain-aws`, `requests`, etc.
- Credenciales AWS (perfil default / entorno de produccion)
- Red hacia Bedrock + Knowledge Bases + API de modelos UFM

---

## 0. Antes de la corrida

1. Revisar si hace falta:
   - `fixtures/r32_apuntes.txt`
   - `fixtures/r35_secuencia.json`
2. Abrir **esta carpeta** como workspace (opcion C / UTF-8).
3. Tener AWS listo (mismo criterio que `IS_TESTING=False` en `motor/model_iacatching.py`).

---

## 1. Ir a scripts

```powershell
cd <ruta>\pruebas_robustez\scripts
```

---

## 2. Smoke check

```powershell
C:\Users\IT-14\anaconda3\python.exe .\smoke_check.py
```

---

## 3. Dry-run (sin Bedrock)

```powershell
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --dry-run
```

Round-robin `cualquiera`: R01?hayek, R02?hazlitt, R03?mises, R06?muso, R07?general, R08?hayek

---

## 4. Ejecutar las 35

```powershell
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py
```

Salidas en `pruebas_robustez/salidas/`.

---

## 5. Opciones utiles

```powershell
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --solo-alta
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --ids R01,R14,R22
C:\Users\IT-14\anaconda3\python.exe .\runner_robustez.py --resume --salida ..\salidas\resultados_chh_XXXX.csv
```

---

## 6. Estructura relevante

```
pruebas_robustez/
  motor/model_iacatching.py     # motor local (system prompts + chains + KB)
  scripts/runner_robustez.py
  scripts/chatbot_client.py     # importa motor local
  bateria_robustez_chh_v1.json  # prompts de prueba
  fixtures/
  salidas/
  referencia_prompts_sistema/   # copia legible de system prompts
```

Si en el futuro actualizas el motor de produccion y quieres alinear esta copia (con el repo padre disponible):

```powershell
copy ..\config\model_iacatching.py .\motor\model_iacatching.py
```

---

## 7. Que entregar

CSV en `salidas/` + `.meta.json`. Sin puntuar pass/fail.
