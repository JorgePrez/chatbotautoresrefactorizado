import random

# === Preguntas por autor ===

hayek_questions = [
    "¿Quién es Friedrich A. Hayek?",
    "¿Por qué es importante conocer la obra de Friedrich A. Hayek?",
    "¿Cuál fue la mayor aportación de Friedrich A. Hayek en economía?",
    "¿Qué es la libertad para Friedrich A. Hayek?",
    "¿Qué es el concepto de 'orden espontáneo' y por qué es fundamental en la filosofía de Hayek?",
    "¿Cuál es la relación entre Friedrich A. Hayek y Ludwig von Mises?",
    "¿Cuál es la relación entre Friedrich A. Hayek y John Maynard Keynes?",
    "¿De qué se trata Los fundamentos de la libertad?",
    "¿Por qué es crucial comprender la diferencia entre legislación y ley según Hayek?",
    "¿Por qué es importante la palabra arbitraria en la definición de libertad de Hayek?",
    "¿Qué libros escribió Friedrich A. Hayek?",
    "¿Por qué un estudiante debería estudiar a Friedrich A. Hayek?",
    "¿Cómo puedo aplicar las ideas de Friedrich A. Hayek en mi vida profesional o académica?",
    "¿Cuáles son las obras principales de Hayek y de qué tratan?",
    "¿Qué implicaciones éticas tienen las advertencias de Hayek sobre la planificación centralizada y la libertad individual?",
    "¿Qué son 'cosmos' y 'taxis' en la teoría de Hayek?",
    "¿Qué son 'nomos' y 'thesis' según Hayek?",
    "¿Por qué ganó Friedrich A. Hayek el Premio Nobel?",
    "¿Qué es la teoría del ciclo económico según Hayek?",
    "¿Cómo aborda Hayek la relación entre las normas, la moral, tradición y evolución de las leyes?"
]

hazlitt_questions = [
    "¿Quién fue Henry Hazlitt?",
    "¿Quién fue Henry Hazlitt y por qué su obra es relevante en el estudio de la economía moderna?",
    "¿Cuál fue el impacto de Economía en una lección en la comprensión pública de la economía y cómo sigue siendo relevante hoy?",
    "¿Cómo define Hazlitt el concepto de consecuencias a corto y largo plazo en las políticas económicas?",
    "¿Qué es el principio de \"coste invisible\" y cómo lo utiliza Hazlitt para criticar la intervención estatal?",
    "¿Cómo explica Hazlitt los efectos de la inflación en La crisis inflacionaria y cómo resolverla?",
    "¿Qué relación tuvo Henry Hazlitt con economistas como Ludwig von Mises y cómo influyó en su pensamiento?",
    "¿En qué aspectos Henry Hazlitt se distancia del keynesianismo y qué críticas fundamentales realiza en Los críticos de la economía keynesiana?",
    "¿Cuál es el papel de la moralidad en la economía según Hazlitt, especialmente en Los fundamentos de la moral?",
    "¿Cómo conecta Hazlitt la libertad individual con el éxito del libre mercado y la prosperidad económica?",
    "¿Por qué Henry Hazlitt critica la planificación centralizada y cuáles son las consecuencias que anticipa para la libertad individual y la economía?",
    "¿Cómo argumenta Hazlitt que el gasto gubernamental afecta negativamente a la eficiencia económica y al bienestar social?",
    "¿Cómo aborda Hazlitt la pobreza en La conquista de la pobreza y qué soluciones propone desde una perspectiva de mercado libre?",
    "¿Qué enseñanzas pueden extraerse de la obra de Hazlitt para enfrentar los desafíos económicos contemporáneos, como la deuda y la inflación?",
    "¿Cómo puede un estudiante aplicar las ideas de Hazlitt en su vida profesional o académica para entender mejor las políticas económicas?",
    "¿Qué aportaciones de Hazlitt siguen siendo cruciales para comprender los debates actuales sobre la política fiscal y monetaria?",
    "¿Quién fue Henry Hazlitt y cuál fue su contribución al periodismo económico?",
    "¿Por qué se considera a Hazlitt como uno de los principales divulgadores de la economía del libre mercado?",
    "¿Cuáles fueron los principales trabajos de Henry Hazlitt, además de Economía en una lección, y qué impacto tuvieron?",
    "¿Cómo contribuyó Hazlitt a la popularización de las ideas de Ludwig von Mises?",
    "¿Qué influencias filosóficas y económicas marcaron el pensamiento de Henry Hazlitt?",
    "¿Cómo se diferencia Hazlitt de otros economistas liberales de su época, como Friedrich Hayek y Milton Friedman?",
    "¿Cómo definió Henry Hazlitt la relación entre la economía y la moralidad en su obra Los fundamentos de la moral?",
    "¿Cómo contribuyó Henry Hazlitt al debate sobre la intervención estatal en la economía?",
    "¿Cómo fue el enfoque de Hazlitt hacia las consecuencias a largo plazo de las políticas económicas, y por qué es importante su perspectiva?",
    "¿Qué relación tuvo Hazlitt con otras figuras relevantes del liberalismo económico, como Ayn Rand, y cómo influyeron en su pensamiento?"
]

mises_questions = [
    "¿Qué es la praxeología según Mises?",
    "¿Cómo define Mises la acción humana?",
    "¿Qué papel juega el cálculo económico en el pensamiento de Mises?",
    "¿Por qué Mises defiende el libre mercado frente al socialismo?",
    "¿Qué crítica hace Mises a la planificación central?",
    "¿Qué entiende Mises por intervencionismo?",
    "¿Cómo explica Mises la función del dinero en la economía?",
    "¿Cuál es la relación entre individuo y sociedad para Mises?",
    "¿Qué opina Mises sobre la inflación y su impacto?",
    "¿Qué dice Mises sobre el conocimiento y los precios?"
]

# === Función para obtener sugerencias ===
def get_sugerencias_por_autor(author: str, cantidad=4):
    if author == "hayek":
        return random.sample(hayek_questions, cantidad)
    elif author == "hazlitt":
        return random.sample(hazlitt_questions, cantidad)
    elif author == "mises":
        return random.sample(mises_questions, cantidad)
    elif author == "general":
        all_suggested_questions = hayek_questions + hazlitt_questions + mises_questions
        shuffled = all_suggested_questions.copy()
        random.shuffle(shuffled)
        return shuffled[:cantidad]
    return []
