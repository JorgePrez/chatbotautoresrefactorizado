# Bloque 2 — Evaluación cuantitativa del motor (sin referencia)
**Evaluación del Chatbot CHH (UFM) · 17 de julio de 2026**
**Base:** núcleo evaluable — 1.518 interacciones de producción (abr–jul 2026) con pregunta + 25 fragmentos recuperados + respuesta.
**Marco:** métricas RAGAS sin referencia (*faithfulness*, *answer relevancy*, *context relevance*) + trazabilidad de citas. Las métricas con referencia (*context recall*, *answer correctness*) requieren el golden set (Bloque 4).

---

## 1. Método en dos capas

Sin acceso a un LLM juez a escala en este entorno, se aplicó un diseño que preserva el rigor y deja lista la escala:

1. **Cribado automático (100% del núcleo):** cada respuesta se segmentó en frases (38.296 en total) y se midió el soporte de cada una contra los 25 fragmentos de su consulta (similitud TF-IDF de caracteres con ventanas). También se midió la relevancia léxica pregunta↔fragmentos y pregunta↔respuesta.
2. **Juicio experto de calibración (pilotaje):** 48 frases estratificadas por nivel de soporte y autor, provenientes de 16 interacciones, verificadas manualmente afirmación por afirmación contra sus 3 mejores fragmentos.

La calibración validó el instrumento: **las 11 afirmaciones no soportadas detectadas tienen todas soporte léxico ≤0,18, mientras las soportadas promedian 0,35**. El cribado ordena bien el riesgo; no es una medida absoluta (las paráfrasis fieles también puntúan bajo), y por eso los porcentajes poblacionales se presentan como banda preliminar.

## 2. Resultados de fidelidad (faithfulness)

**Pilotaje juzgado (37 afirmaciones factuales de 16 interacciones):**

| Veredicto | n | % |
|---|---|---|
| Soportada por los fragmentos (incl. paráfrasis fiel) | 19 | 51% |
| Parcial (soportado + añadido) | 7 | 19% |
| **No soportada por los fragmentos** | **11** | **30%** |

A nivel de interacción: **7 de 16 conversaciones juzgadas contenían al menos una afirmación no fundamentada** en el contexto recuperado.

**Extrapolación poblacional (preliminar):** el 50% de las frases del núcleo cae bajo el umbral de riesgo; aplicando la tasa de confirmación del pilotaje, la banda estimada de afirmaciones no fundamentadas es del **15–30% de las frases factuales**. Es una banda ancha por diseño — el número definitivo saldrá del juez LLM a escala (script entregado, ver §6) — pero suficiente para afirmar que **el fenómeno no es marginal**.

Matiz importante: "no soportada" ≠ "falsa". Varias afirmaciones no fundamentadas son correctas sobre el autor (el modelo las conoce por su entrenamiento). El problema es de **diseño y de confianza**: el sistema promete responder "únicamente con la información recuperada" y no lo cumple de forma consistente — y cuando el modelo rellena por su cuenta, nada garantiza que siga acertando.

## 3. El mecanismo del fallo: recuperación irrelevante → generación sin ancla

El hallazgo más accionable del bloque. Las interacciones con afirmaciones no soportadas tienen **la mitad de relevancia de contexto** que las limpias (0,18 vs 0,39 de solape pregunta-fragmentos). El patrón observado en los casos verificados:

- **"¿Cómo define Hayek la mente?"** → la recuperación devuelve material genérico sobre "por qué estudiar a Hayek"; el bot responde con contenido de *The Sensory Order* **que no está en los fragmentos** (correcto académicamente, no fundamentado en el corpus).
- **"¿Qué escribo sobre la Escuela Austriaca?"** (chatbot hazlitt) → la recuperación devuelve fragmentos de *La sabiduría de los estoicos*; el bot recomienda bibliografía desde su conocimiento paramétrico.
- **"¿Qué opina Muso de Árbenz y Arévalo?"** → el bot, **hablando en primera persona como Ayau**, emite juicios políticos ("Ni Arévalo ni Árbenz eran monstruos", "algunas reformas fueron necesarias") **sin soporte alguno en los fragmentos**. Es el caso más delicado: opiniones políticas atribuidas a una persona real fallecida sin fuente. Debe tratarse como prioridad con el CHH.

En cambio, cuando la recuperación acierta, la fidelidad observada es alta, con paráfrasis fiel y citas verbatim correctas (los conceptos económicos nucleares — interés, período de producción, fundación de la UFM, política monetaria en Hazlitt — salieron impecables en el pilotaje).

**Implicación de diseño:** el punto débil no es la generación sino el eslabón recuperación→generación. Dos palancas concretas a probar: (a) umbral de score o reranking para no enviar 25 fragmentos indiscriminados (p10 de score: 0,44), y (b) instrucción de abstención explícita cuando el material recuperado no responde a la pregunta ("no tengo ese contenido en mi corpus"), en lugar de rellenar.

## 4. Trazabilidad de citas textuales

Sobre las 251 citas presentadas como textuales (formato blockquote o comillas) en el núcleo: **el 57% se localiza verbatim en los fragmentos de esa consulta; el 43% no**. Dos caveats: en conversaciones multi-turno la cita puede provenir de fragmentos de turnos anteriores (no unidos en esta versión del dataset), y parte de los fallos son paráfrasis didácticas **presentadas con formato de cita literal** — lo cual es en sí un problema de integridad de citación en un producto académico. Por autor, hayek traza mejor (62%) y muso/mises peor (≈0% en muestras pequeñas). Métrica a refinar con el juez LLM incorporando el contexto conversacional completo.

## 5. Los rechazos de muso, explicados

Los 26 rechazos de muso en el núcleo **no se explican por recuperación fallida** (su relevancia de contexto media es 0,51, superior a la media). Son mayoritariamente respuestas de alcance ante preguntas biográficas/personales o peticiones técnicas (generar PDF, material de exámenes). Es decir: el persona-prompt de muso es más conservador, no peor recuperador. Queda la paradoja de que **el mismo bot que rechaza por prudencia es el que opina de Árbenz sin fuente** — el criterio de abstención existe pero no se activa donde más falta hace.

## 6. Entregables y siguiente paso de escala

- `ragas_dataset.jsonl` — las 1.392 interacciones en **formato RAGAS estándar** (`question`, `answer`, `contexts`), listo para la librería RAGAS o cualquier juez.
- `judge_faithfulness.py` — juez LLM afirmación-por-afirmación (definición RAGAS de faithfulness + alertas de citas/atribuciones), reanudable, con coste estimado: ~$20 una muestra de 300, ~$100 el núcleo completo. Ejecutable por vuestro equipo con clave propia.
- `nucleo_screening.parquet` y `frases_flagged.parquet` — ranking de riesgo de las 1.518 interacciones y las 19.010 frases señaladas, para revisión priorizada (el top-150 de riesgo: 69 hayek, 53 general, 18 muso).
- `calibracion_juzgada.json` — las 48 frases del pilotaje con veredicto experto, reutilizables para validar el juez LLM (meta-evaluación).

## 7. Conclusiones provisionales del motor

1. **Cuando la recuperación funciona, el motor es fiel** — paráfrasis correcta y citas verbatim exactas en los conceptos nucleares del corpus.
2. **Cuando la recuperación falla, el motor no se abstiene: rellena** desde conocimiento paramétrico. Banda preliminar de afirmaciones no fundamentadas: 15–30%. Es el hallazgo central a confirmar con el juez a escala y a atacar con umbral/reranking + instrucción de abstención.
3. **Riesgo reputacional concreto:** opiniones políticas en primera persona atribuidas a Ayau sin fuente. Revisar con el CHH antes que cualquier otra mejora.
4. **Integridad de citación mejorable:** una parte relevante de lo presentado como cita literal no lo es.
5. Estos resultados delimitan exactamente lo que el golden set debe medir en el Bloque 4: corrección factual de esa zona de relleno, y el criterio de los profesores sobre qué debe responderse desde el corpus y qué debe rechazarse.
