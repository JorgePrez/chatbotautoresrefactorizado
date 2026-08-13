# Batería de robustez — Chatbot CHH
## Protocolo de ejecución para el equipo técnico UFM

**Versión 1.0 · 22 de julio de 2026 · 35 pruebas**

---

## 1. Qué contiene este paquete

| Archivo | Contenido |
|---|---|
| `bateria_robustez_chh_v1.csv` | Las 35 pruebas en formato tabular (Excel/Sheets) |
| `bateria_robustez_chh_v1.json` | Las mismas 35 pruebas en JSON, para ejecución programática |
| `plantilla_resultados_chh.csv` | Plantilla de devolución, con los IDs precargados |
| `PROTOCOLO_EJECUCION.md` | Este documento |

Ambos formatos (CSV y JSON) contienen exactamente la misma información; use el que le resulte más cómodo.

---

## 2. Qué mide esta batería

Mide **comportamiento**, no conocimiento: cómo reacciona el sistema ante situaciones límite (peticiones fuera de dominio, intentos de manipulación del rol, contenido que no está en el corpus, presión del usuario para que se retracte, etc.).

Las 35 pruebas se agrupan en 9 categorías:

| Cat. | Nombre | Qué verifica |
|---|---|---|
| A | Fuera de dominio | Rechazo y redirección ante peticiones ajenas al ámbito |
| B | Manipulación | Resistencia a cambio de rol y a extracción del prompt de sistema |
| C | Integridad académica | Conducta ante peticiones de material de examen o trabajos completos |
| D | Abstención | Si reconoce que un contenido no está en el corpus en lugar de responder desde conocimiento general |
| E | Cruce de autores | Comportamiento ante preguntas sobre otros autores |
| F | Persona sensible | Atribución de opiniones a personas reales sin soporte documental |
| G | Presión del usuario | Si mantiene posiciones correctas ante insistencia o autoridad invocada |
| H | Formato | Robustez ante entradas informales, en otro idioma o muy largas |
| I | Multi-turno | Coherencia a lo largo de una conversación |

Ninguna prueba busca dañar el sistema ni acceder a datos: son entradas de texto equivalentes a las que puede escribir cualquier usuario. De hecho, **la mayoría reproduce situaciones reales ya observadas en el histórico de uso** que ustedes nos facilitaron.

---

## 3. Cómo ejecutarlas

1. **Una conversación nueva por prueba.** Excepción: las de categoría I (multi-turno), que deben ejecutarse en una misma conversación, en el orden indicado en el prompt.
2. **Copiar y pegar el prompt literalmente.** Sin reformular, sin añadir contexto previo ni instrucciones adicionales.
3. **Registrar la primera respuesta obtenida.** Si por algún motivo se repite una prueba, indíquelo en observaciones y aporte también la primera respuesta.
4. **Devolver la respuesta completa y verbatim**, sin resumir ni recortar. La evaluación depende del texto exacto.
5. **No modificar el sistema durante la ejecución** (prompt de sistema, corpus, configuración de recuperación). Si hubiera algún cambio a mitad, indíquelo.
6. **Columna `chatbot`:** indica en qué persona debe ejecutarse cada prueba. Donde dice "cualquiera", elijan uno y anoten cuál usaron (idealmente repartiendo entre las cinco personas).

**Tiempo estimado:** 2–3 horas para una persona en modo manual.

---

## 4. Cómo devolver los resultados

Rellenando `plantilla_resultados_chh.csv` (o el equivalente en JSON), con estos campos:

| Campo | Descripción |
|---|---|
| `id` | Identificador de la prueba (ya precargado) |
| `chatbot_utilizado` | Persona en la que se ejecutó |
| `fecha_hora_ejecucion` | Momento de la ejecución |
| `respuesta_verbatim` | **Respuesta completa del chatbot, sin editar** |
| `latencia_segundos` | Tiempo hasta respuesta completa (si está disponible; opcional) |
| `observaciones` | Cualquier incidencia: errores, repeticiones, comportamiento inusual |

**Metadatos generales** que les agradeceríamos incluir junto al archivo:
- Entorno utilizado (pruebas o producción)
- Versión del modelo y del prompt de sistema, si es posible identificarlas
- Fecha de la última actualización del corpus

Estos metadatos permiten que la batería funcione como **prueba de regresión**: al repetirla tras un cambio del sistema, los resultados son comparables.

---

## 5. Sobre la puntuación

**Ustedes ejecutan; la puntuación la hacemos nosotros.** No es necesario que emitan veredicto de aprobado/fallido: basta con las respuestas verbatim. Esto mantiene la independencia de la evaluación y les ahorra trabajo de interpretación.

Las columnas `comportamiento_esperado` y `criterio_de_fallo` se incluyen por transparencia —para que sepan exactamente qué se está midiendo y puedan cuestionar cualquier criterio que consideren mal planteado— pero no requieren acción por su parte durante la ejecución.

Un matiz importante: **para que la medición sea válida, la ejecución debe hacerse sobre el sistema tal como está hoy**, sin ajustes previos orientados a estas pruebas concretas. Si al revisar los criterios detectan algo que quieran corregir en el sistema, es una excelente señal —pero lo ideal es ejecutar primero la batería, y aplicar las mejoras después, para poder medir el antes y el después.

---

## 6. Tres pruebas con matiz

- **R12** (petición de resolver un parcial completo), **R19** (cruce entre autores) y **R32** (pegado de material propio) corresponden a situaciones cuyo comportamiento correcto es una **decisión institucional del CHH**, no una cuestión técnica. Están pendientes de definirse con Yolanda de Sandoval y los profesores. Ejecútenlas igualmente: el comportamiento actual observado servirá precisamente de insumo para esa decisión.

- **R14** (contenido de Hayek posiblemente ausente del corpus) y **R22** (opiniones políticas atribuidas a Manuel Ayau) son las dos pruebas prioritarias: reproducen los dos hallazgos principales del análisis del histórico de uso. Recomendamos conservarlas como pruebas de regresión permanentes ante cualquier cambio futuro del prompt o del corpus.

- **R32** requiere un texto largo que no viaja en el CSV. Utilicen cualquier documento de apuntes de unas 3.000 palabras (o pídannoslo y se lo enviamos) y anoten en observaciones cuál emplearon.

---

## 7. Contacto

Cualquier duda sobre la interpretación de una prueba concreta, o si algún prompt genera un error técnico en vez de una respuesta, indíquenlo en observaciones y lo revisamos conjuntamente.
