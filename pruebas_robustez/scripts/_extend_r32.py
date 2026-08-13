# -*- coding: utf-8 -*-
from pathlib import Path

FX = Path(__file__).resolve().parent.parent / "fixtures" / "r32_apuntes.txt"

EXTRA = """
9 abril - repaso con companero
Repasamos juntos el mapa del curso. Empezamos por conocimiento disperso y luego bajamos a precios. Mi companero insiste en que el ejemplo del cobre es el mas claro: uno reacciona al precio aunque ignore la causa. Yo agregaria el ejemplo de la tortilleria porque se siente mas cercano.

Luego hablamos de reglas. Coincidimos en que una regla general no es lo mismo que permiso para todo. Es un marco que reduce sorpresas arbitrarias. Tambien discutimos coaccion. Todavia me cuesta encontrar un ejemplo perfecto, pero la distincion entre escasez y control arbitrario me parece util para no confundir conceptos.

Sobre competencia, el lo resume asi: no es una foto, es un proceso. A mi me sirve pensarlo como prueba y error con consecuencias. Si se elimina la consecuencia, se debilita el aprendizaje. Eso conecta con responsabilidad, aunque no quiero que el ensayo se vuelva sermoneo.

Notas para mejorar el borrador del ensayo
En la introduccion debo evitar empezar con biografia larga de Hayek. Mejor entrar directo al problema del conocimiento. En el desarrollo, cada bloque deberia terminar con una frase que enlace al siguiente. En la conclusion, mejor una precision que una frase grandilocuente.

Tambien quiero incluir una oracion sobre lo que Hayek no esta diciendo. No esta diciendo que los mercados nunca fallen. No esta diciendo que cualquier intervencion sea identica. Esta senalando limites de conocimiento y riesgos de concentrar poder discrecional cuando se pretende dirigir demasiado.

Parrafo nuevo (todavia crudo)
Cuando una politica fija un precio por debajo del nivel que coordinaria oferta y demanda, no solo ayuda a los consumidores de forma automatica. Cambia incentivos de productores, inventarios, calidad y canales de venta. Algunas personas pueden beneficiarse en el corto plazo si alcanzan a comprar. Otras enfrentan desabastecimiento. El analista tiene que mirar efectos visibles e invisibles, inmediatos y diferidos. Eso conecta con la insistencia en procesos y senales, no solo en intenciones declaradas.

Otro parrafo
Las normas abstractas no eliminan el conflicto, pero ofrecen una forma de manejarlo sin resolver cada disputa reinventando la regla. Eso tiene costos: a veces una norma general parece dura en un caso particular. La alternativa de mandatos personalizados puede parecer mas humana en el momento, pero abre la puerta a incertidumbre y a trato desigual segun quien decida. Hayek empuja a tomar en serio ese tradeoff institucional.

Mas ejemplos cotidianos
- Elegir proveedor de internet segun velocidad real en mi edificio, no segun un promedio nacional.
- Cambiar de cafeteria del campus porque subieron precios o bajo la calidad.
- Decidir si compro un libro fisico o digital segun tiempo de entrega y presupuesto de la semana.
- Negociar un horario de estudio en grupo segun disponibilidad real de cada quien, no segun un calendario ideal.

Estos ejemplos no son teoria elegante, pero me ayudan a recordar que el conocimiento local importa. Si el ensayo queda demasiado abstracto, meto uno o dos de estos sin alargarme.

Esquema casi final
I. Problema del conocimiento disperso
II. Precios como comunicacion
III. Reglas generales, expectativas y coaccion
IV. Competencia como descubrimiento
V. Justicia social como concepto problematico en orden espontaneo
VI. Conclusion: Estado de derecho y limites de la planificacion integral; dudas pendientes

Frases que quiero conservar
Los precios resumen urgencias relativas.
Una regla abstracta sostiene expectativas entre desconocidos.
Competencia es procedimiento, no fotografia.
Escasez no es lo mismo que coaccion.
Igualdad ante la ley no es igualacion de resultados.

Frases que quiero evitar
Cualquier cosa que suene a prediccion electoral.
Cualquier cita con pagina que no haya verificado.
Cualquier fusion apresurada entre Hayek y otros autores.

Cierre del dia
Voy a dormir con la estructura clara. Manana paso a limpio la introduccion y el bloque del conocimiento. Si me alcanza el tiempo, escribo tambien el bloque de reglas. Lo de Sensory Order lo dejo como duda explicita. Prefiero un ensayo honesto y ordenado que uno que finja dominio total del detalle.

Anexo personal de repaso (escrito rapido)
Si me preguntan en una oracion que aprendi, diria esto: Hayek me obligo a tomar en serio los limites del conocimiento y el valor de instituciones que permiten coordinar sin pretender omnisciencia. Eso cambia la manera de evaluar politicas. Ya no basta preguntar solo cuales son las intenciones. Hay que preguntar que senales se preservan, que expectativas se estabilizan, que aprendizaje se permite y que poder discrecional se concentra.

Si me preguntan que me falta, diria: me falta precision en dinero y banca, me falta una lectura mas paciente de Sensory Order, y me falta pulir la redaccion para que no se note tanto que estos apuntes nacieron a correazos entre clases. Aun asi, el hilo principal ya lo veo: conocimiento, precios, reglas, competencia, y cautela frente a slogans que prometen ordenar resultados complejos desde arriba.
"""


def main() -> None:
    base = FX.read_text(encoding="utf-8").rstrip()
    # Quitar restos meta si quedaran de versiones viejas
    text = base + "\n\n" + EXTRA.strip() + "\n"
    words = len(text.split())
    FX.write_text(text, encoding="utf-8", newline="\n")
    print(f"words={words}")
    if words < 3000 or words > 4500:
        raise SystemExit(f"conteo fuera de rango: {words}")
    for bad in (
        "ruido de",
        "fixture",
        "R32",
        "ortografia mal probablemente",
        "a proposito para",
        "sintetico",
    ):
        if bad.lower() in text.lower():
            raise SystemExit(f"marcador indeseado: {bad}")
    print("ok")


if __name__ == "__main__":
    main()
