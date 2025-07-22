import random

# === Preguntas por autor, primarias entre 25 y 30 ===

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

mises_questions= [
"¿Quién fue Ludwig von Mises y cuál fue su influencia en la Escuela Austriaca de economía?",
"¿Por qué es importante estudiar a Ludwig von Mises hoy en día?",
"¿Qué es la praxeología y por qué es fundamental en la metodología de Mises?",
"¿Qué papel cumple la libertad individual en el pensamiento de Mises?",
"¿Cuál es la relación intelectual entre Ludwig von Mises y Friedrich A. Hayek?",
"¿Qué es el cálculo económico y cómo lo usa Mises para refutar el socialismo?",
"¿Por qué sostiene Mises que el socialismo es inviable como sistema económico?",
"¿Qué significa “acción humana” en su sistema teórico?",
"¿Qué libros fundamentales escribió Ludwig von Mises?",
"¿Cómo se relacionan economía, lógica y ética en el pensamiento de Mises?",
 "¿En qué contexto histórico se formó Mises como economista?",
"¿Cómo influyó su experiencia en Austria y su exilio en Estados Unidos en su obra?",
"¿Por qué Mises rechaza el empirismo y el positivismo en economía?",
"¿Qué diferencia hay entre praxeología y psicología en la explicación de la acción humana?",
"¿Cómo define Mises la relación entre mercado y cálculo económico?",
"¿Qué papel tiene el dinero en La teoría del dinero y del crédito?",
"¿Qué crítica hace Mises al intervencionismo?",
"¿Por qué Mises sostiene que sin propiedad privada no puede haber libertad ni racionalidad económica?",
"¿Cómo responde Mises a los argumentos del historicismo y el relativismo?",
"¿Qué distingue el liberalismo de Mises del liberalismo contemporáneo?",
"¿Cómo se puede aplicar la praxeología al análisis de políticas públicas?",
"¿Qué aportes hace Mises al debate sobre inflación y banca central?",
"¿Cuáles son las consecuencias sociales de la inflación según Mises? ",
"¿Cómo influye la teoría subjetiva del valor en la obra de Mises?",
"¿Qué ejemplos históricos ilustran la crítica misesiana al socialismo?", 
"¿Cuál es el papel del empresario en la economía de mercado según Mises?", 
"¿Cómo se relacionan la moral y la acción humana en su teoría?",
"¿Qué críticas ha recibido la praxeología y cómo respondió Mises a ellas?", 
"¿Dónde enseñó y difundió sus ideas Ludwig von Mises durante su vida?",
"¿Cómo influyó Mises en el pensamiento liberal del siglo XX? "
]




muso_questions = [
    "¿Quién fue Manuel F. Ayau y cuál fue su papel en la vida pública guatemalteca?",
    "¿Qué es el liberalismo clásico y cómo lo defendía?",
    "¿Por qué consideraba esencial la libertad individual para el progreso económico?",
    "¿Cuándo y cómo fundé el CEES?",
    "¿Dónde se originó mi preocupación por el subdesarrollo?",
    "¿Qué principios éticos guían el pensamiento económico?",
    "¿Cómo explico el origen de la riqueza?",
    "¿Cuál es la función del capital?",
    "¿Por qué criticaba el salario mínimo?",
    "¿Qué relación hay entre productividad y nivel de vida?",
    "¿Cómo entiendo el rol de los precios en una economía libre?",
    "¿Qué argumentos doy contra el impuesto progresivo?",
    "¿Qué es el orden espontáneo y por qué es esencial en mi teoría?",
    "¿Cómo justifico la propiedad privada desde lo ético y lo práctico?",
    "¿Por qué fundé la Universidad Francisco Marroquín?",
    "¿Qué consecuencias económicas tiene la indemnización por despido según Ayau?",
    "¿Por qué decía que un aumento de sueldos sin productividad no mejora el bienestar?",
    "¿Cómo argumenta que los impuestos progresivos perjudican más a los pobres?",
    "¿Qué significa que el capital es “trabajo no consumido”?",
    "¿Cómo critico la redistribución forzada de la riqueza?",
    "¿Qué rol cumple el ahorro en la formación de capital?",
    "¿Cómo diferencio entre \"riqueza productiva\" y \"riqueza aparente\"?",
    "¿Qué consecuencias tiene el proteccionismo para las exportaciones?",
    "¿Cómo afecta el salario mínimo a los trabajadores menos productivos?",
    "¿Por qué considero la intervención estatal como una fuente de pobreza?",
    "¿Cuándo afirmo que una sociedad se empobrece moral y materialmente?",
    "¿Dónde y por qué han fracasado históricamente las políticas colectivistas?",
    "¿Qué papel atribuyo al sistema jurídico en el desarrollo económico?",
    "¿Cómo vinculo la ignorancia económica con la acción política?",
    "¿Qué lecciones saco de la historia del socialismo y el mercantilismo?",
        "¿Cómo se puede aplicar hoy la \"ley de asociación\" que Ayau explica en su ensayo sobre Robinson y Viernes?",
    "¿Qué enseñanzas de Ayau ayudarían a combatir la pobreza en comunidades rurales?",
    "¿Cómo afectaría al sistema educativo una reforma inspirada en los principios de Ayau?",
    "¿Qué políticas actuales contradicen directamente las ideas de Ayau?",
    "¿Dónde podría verse hoy la paradoja de los subsidios que él criticaba?",
    "¿Cuándo sería moralmente legítima una intervención estatal, si seguimos el pensamiento de Ayau?",
    "¿Qué rol deben tener los empresarios en una sociedad libre según Ayau?",
    "¿Por qué Ayau advertía sobre el peligro de confundir redistribución con justicia?",
    "¿Cómo se compara el enfoque de Ayau con el de economistas como Keynes?",
    "¿Qué desafíos enfrenta la libertad económica en América Latina según el pensamiento de Ayau?",
    "¿Por qué Ayau afirmaba que el colectivismo “empobrece al pobre”?",
    "¿Cómo argumentaría Ayau contra una política de precios tope?",
    "¿Cuáles son los efectos culturales del estatismo que señala Ayau?",
    "¿Qué tipo de ciudadanía necesita una sociedad libre, según Ayau?",
    "¿Cómo utilizar el pensamiento de Ayau para evaluar propuestas de reforma constitucional?",
    "¿Qué principios deberían regir la cooperación internacional desde la perspectiva de Ayau?",
    "¿Cómo concibe Ayau el papel de las universidades en una sociedad libre?",
    "¿Qué significa “crear riqueza sin quitársela a nadie”?",
    "¿Qué preguntas haría Ayau sobre una nueva política fiscal?",
    "¿Dónde están las principales barreras al desarrollo en su análisis de Guatemala?",
    "¿Cómo se construye un mercado verdaderamente libre, según Ayau?",
    "¿Qué implicaciones éticas tiene la defensa de la libertad en su pensamiento?",
    "¿Por qué afirmaba que el Estado no es fuente de riqueza?",
    "¿Qué enseñanzas de Ayau son clave para la formación de futuros líderes?",
    "¿Cómo se traduce el \"sentido común económico\" de Ayau en decisiones cotidianas?"
]



muso_questions_mencion_ayau = [
    "¿Quién fue Manuel F. Ayau y cuál fue su papel en la vida pública guatemalteca?",
    "¿Qué es el liberalismo clásico según Manuel F. Ayau y cómo lo defendía?",
    "¿Por qué consideraba Manuel F. Ayau que la libertad individual es esencial para el progreso económico?",
    "¿Qué principios éticos guiaban el pensamiento económico de Manuel F. Ayau?",
    "¿Cómo explicaba Manuel F. Ayau el origen de la riqueza?",
    "¿Cuál es la función del capital según Manuel F. Ayau?",
    "¿Por qué criticaba Manuel F. Ayau el salario mínimo?",
    "¿Qué relación establecía Manuel F. Ayau entre productividad y nivel de vida?",
    "¿Cómo entendía Manuel F. Ayau el rol de los precios en una economía libre?",
    "¿Qué argumentos daba Manuel F. Ayau contra el impuesto progresivo?",
    "¿Qué es el orden espontáneo y por qué era esencial en la teoría de Manuel F. Ayau?",
    "¿Cómo justificaba Manuel F. Ayau la propiedad privada desde lo ético y lo práctico?",
    "¿Por qué fundó Manuel F. Ayau la Universidad Francisco Marroquín?",
    "¿Por qué consideraba Manuel F. Ayau que la intervención estatal es una fuente de pobreza?",
    "¿Cómo criticaba Manuel F. Ayau la redistribución forzada de la riqueza?"
]

# === Función para obtener sugerencias ===
def get_sugerencias_por_autor(author: str, cantidad=4):
    if author == "hayek":
        return random.sample(hayek_questions, cantidad)
    elif author == "hazlitt":
        return random.sample(hazlitt_questions, cantidad)
    elif author == "mises":
        return random.sample(mises_questions, cantidad)
    elif author == "muso":
        return random.sample(muso_questions, cantidad)
    elif author == "general":
        all_suggested_questions = hayek_questions + hazlitt_questions + mises_questions + muso_questions_mencion_ayau
        shuffled = all_suggested_questions.copy()
        random.shuffle(shuffled)
        return shuffled[:cantidad]
    return []
