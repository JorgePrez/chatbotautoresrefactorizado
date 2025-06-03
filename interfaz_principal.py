import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from config.dynamo_crud import getChats
import uuid
from config import dynamo_crud as DynamoDatabase



# Configuración inicial
st.set_page_config(page_title="Interfaz Principal",layout="wide")


# Cargar configuración del archivo YAML
with open('userschh_login_google.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Inicializar autenticador
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)


st.markdown("""
    <style>
    /* Eliminar margen superior general */
    html, body, [data-testid="stAppViewContainer"] {
        margin: 0;
        padding: 0;
        height: 100%;
    }

    /* Eliminar el espacio del header */
    [data-testid="stHeader"] {
        display: none;
    }

    /* Opcional: eliminar espacio adicional del main container */
    .block-container {
        padding-top: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    .user-info {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .user-info img {
        border-radius: 50%;
        height: 36px;
    }

    .user-info p {
        margin: 0;
        font-weight: bold;
    }

    .main-content-wrapper {
        max-width: 700px;
        margin: 0 auto;
        text-align: center;
    }

    .input-wrapper {
        max-width: 700px;
        margin: 10px auto 0 auto;
    }

    .button-row-centered {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
            
    .stButton > button {
    border-radius: 25px;
    padding: 0.5rem 1.5rem;
    border: 1.5px solid #d60812 !important;
    color: black;
    background-color: white;
    transition: all 0.3s ease;
}

    .stButton > button:hover {
    background-color: #d60812;
    color: white;
}

    .stTextInput input {
        border: 1.5px solid #d60812 !important;
        border-radius: 10px;
        padding: 0.5rem;
    }
            
    .titulo-central {
    text-align: center;
    margin-bottom: 0.5rem;
    }

    .subtitulo-central {
        margin-top: 0;
        text-align: center
    }

    .fade-in {
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .texto-descriptivo {
        margin-top: 30px;
        font-size: 0.95rem;
        text-align: center;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
        
    [data-testid="stHeader"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        height: 0px !important;
    }
            
</style>
""", unsafe_allow_html=True)


#--- Para el DIALOG


# --- CSS personalizado para el dialog---
st.markdown("""
<style>
/* Ampliar el diálogo */
div[data-testid="stDialog"] div[role="dialog"] {
    width: 90vw !important;
    max-width: 550px !important;
}

/* Botón flotante */
.boton-historial {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
}
.boton-historial button {
    background: transparent !important;
    border: none !important;
    border-radius: 50%;
    padding: 8px 10px;
    cursor: pointer;
    font-size: 20px;
    transition: background 0.2s ease;
}
.boton-historial button:hover {
    background-color: rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# Nueva conversación

st.markdown("""
<style>
/* Estilo solo para botones cuyo key comience con 'ir_' (Nueva conversación) */
div[data-testid="stDialog"] [class*="st-key-ir_"] button {
    border-radius: 10px;
    border: 1.5px solid #d6081f !important;
    background-color: white;
    color: black !important;  
    font-size: 14px;
    padding: 6px 14px;
    transition: all 0.3s ease;
    font-weight: 500;
}

/* Hover con inversión */
div[data-testid="stDialog"] [class*="st-key-ir_"] button:hover {
    background-color: #d6081f;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


#Botones conversacion

st.markdown("""
<style>
/* Estilo solo para los botones 'Abrir conversación' */
div[data-testid="stDialog"] [class*="st-key-open_"] button {
    border: none !important;
    background: none !important;
    color: #d6081f !important; /* ← Aquí le das color al ícono */
    font-size: 18px;
    padding: 4px;
    box-shadow: none !important;
}

/* Opcional: cambia color en hover */
div[data-testid="stDialog"] [class*="st-key-open_"] button:hover {
    background-color: rgba(0,0,0,0.05);
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)



#    border-radius: 8px !important;  /* Más cuadrado */

##Para el boton historial


st.markdown("""
<style>
/* Estilo específico para el botón con key="btn_historial" */
div[class*="st-key-btn_historial"] button {
    /*border-radius: 25px;*/
    border-radius: 8px !important;  /* Más cuadrado */
    border: 1.5px solid #d6081f !important;
    background-color: white !important;
    color: black !important;
    font-size: 16px;
    padding: 6px 14px;
    transition: all 0.3s ease;
}

div[class*="st-key-btn_historial"] button:hover {
    background-color: #d6081f !important;
    color: white !important;
}
            

/* SOLO PARA PANTALLAS GRANDES (como escritorio 1920px o más) */
@media (min-width: 1600px) {
    div[class*="st-key-btn_historial"] button {
        margin-left: 20px;
    }
}
</style>
        
</style>
""", unsafe_allow_html=True)


##Para logout
# margin-left: 6px;
st.markdown("""
<style>
/* Estilo específico para el botón con key="btn_propio_logout" */
div[class*="st-key-btn_propio_logout"] button {
    /*border-radius: 25px;*/
    border-radius: 8px !important;  /* Más cuadrado */
    border: 1.5px solid #d6081f !important;
    background-color: white !important;
    color: black !important;
    font-size: 16px;
    padding: 6px 14px;
    transition: all 0.3s ease;
}

div[class*="st-key-btn_propio_logout"] button:hover {
    background-color: #d6081f !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


AUTORES_CONFIG = [
    {
        "label": "Friedrich A. Hayek",
        "key": "hayek",
        "logo": "https://intranet.ufm.edu/reportesai/img_chatbot/hayek_full-noblank.png",
        "avatar_size": 32,
        "pagina": "hayek"
    },
    {
        "label": "Henry Hazlitt",
        "key": "hazlitt",
        "logo": "https://intranet.ufm.edu/reportesai/img_chatbot/Henry-Hazlitt-noblank.png",
        "avatar_size": 34,
        "pagina": "hazlitt"
    },
    {
        "label": "Ludwig von Mises",
        "key": "mises",
        "logo": "https://intranet.ufm.edu/reportesai/img_chatbot/Mises-noblank.png",
        "avatar_size": 32,
        "pagina": "mises"
    },
    {
        "label": "Todos los autores",
        "key": "general",
        "logo": "",
        "avatar_size": 20,
        "pagina": "todos_autores"
    }
]


@st.dialog("🕘 Historial de conversaciones")
def mostrar_historial():
    usuario = st.session_state.get("username", "")
    
    st.markdown("""
    <hr style='border: none; height: 1px; background-color: #d6081f; margin: 8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    for autor in AUTORES_CONFIG:
        conversaciones = getChats(usuario, autor["key"])

        with st.container():
            col_a, col_b = st.columns([1, 1.5])
            with col_a:
                if autor["logo"]:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-weight: 600; font-size: 18px;">{autor['label']}</span>
                            <img src="{autor['logo']}" height="{autor['avatar_size']}" style="border-radius: 50%;">
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-weight: 600; font-size: 18px;">{autor['label']}</span>
                            <span style="font-size: 20px;">🧾</span>
                        </div>
                    """, unsafe_allow_html=True)

            with col_b:
                if st.button("Nueva conversación", icon=":material/open_in_new:", key=f"ir_{autor['label']}", use_container_width=True):
                    nuevo_id = str(uuid.uuid4())
                    st.session_state[f"chat_id_{autor['key']}"] = nuevo_id
                    st.session_state[f"messages_{autor['key']}"] = []
                    st.session_state["autor_a_redirigir"] = autor["key"]
                    st.session_state["cargar_chat_especifico"] = True
                    st.session_state["redirigir_forzado"] = True

                    # Guardar la conversación vacía en la base
                    DynamoDatabase.save(
                        nuevo_id,
                        usuario,
                        autor["key"],
                        "nuevo chat",
                        []
                    )

        if not conversaciones:
            st.markdown(f"<span style='color: #555;'>No tienes conversaciones con <strong>{autor['label']}</strong>.</span>", unsafe_allow_html=True)
        else:
            for item in conversaciones:
                chat_id = item["SK"].split("#")[1]
                nombre = item.get("Name", "Sin título")

                c1, c2 = st.columns([7.5, 0.5])
                c1.markdown(nombre)

                if c2.button("", icon=":material/launch:", key=f"open_{chat_id}", help="Abrir esta conversación",type="tertiary", use_container_width=True):
                    st.session_state[f"chat_id_{autor['key']}"] = chat_id
                    st.session_state["autor_a_redirigir"] = autor["key"]
                    st.session_state["cargar_chat_especifico"] = True
                    st.session_state["redirigir_forzado"] = True
                
             

        st.markdown("""
        <hr style='border: none; height: 1px; background-color: #d6081f; margin: 8px 0 16px 0;'>
        """, unsafe_allow_html=True)

        if st.session_state.get("redirigir_forzado"):
            autor = st.session_state["autor_a_redirigir"]
            st.session_state["redirigir_forzado"] = False  # Reset
            st.switch_page(f"pages/{autor}.py")




# --- Header superior: logo izquierda y login derecha ---
col_logo, col_spacer, col_login = st.columns([2, 6, 2], gap="medium")

#with col_logo:
#    st.markdown(
#        "<div style='padding-top: 10px; padding-left: 100px;'>"
#        "<img src='https://intranet.ufm.edu/reportesai/img_chatbot/LOGO_UFM_FullCol.png' width='150'/>"
#        "</div>",
#        unsafe_allow_html=True
#    )

with col_logo:

    # max-width: 250px;
    st.markdown(
        """
        <div style='padding-top: 10px; padding-left: 100px; max-width: 250px;'>
            <img src='https://intranet.ufm.edu/reportesai/img_chatbot/LOGO_UFM_FullCol.png'
                 style='width: 100%; height: auto;' />
        </div>
        """,
        unsafe_allow_html=True
    )

with col_login:
    st.markdown("<div style='padding-top: 25px;'>", unsafe_allow_html=True)

    if st.session_state.get("authentication_status"):
        username = st.session_state.get("username")
        user_data = config['credentials']['usernames'].get(username, {})
        profile_pic_url = user_data.get("picture", "")

        col1, col2, col3 = st.columns([0.2, 0.3, 0.5]) # 0.2,0.2,0.6
        # [0.2, 0.3, 0.5])
        with col1:
            if profile_pic_url:
                correo = user_data.get("email", "Correo no disponible")
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; align-items: center;'>
                        <img src="{profile_pic_url}" alt=""
                            title="{correo}"
                            onerror="this.src='https://www.gravatar.com/avatar/?d=mp&f=y';"
                            style="width: 36px; height: 36px; object-fit: cover; border-radius: 50%; margin-top: 4px; " />
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            
            if st.button("", icon=":material/description:", key="btn_historial", type="tertiary",  help="Historial de conversaciones"):
                mostrar_historial()


        with col3:
            logout_button = st.button("", key="btn_propio_logout", icon=":material/logout:", help="Cerrar sesión")
            
            if logout_button:
                authenticator.logout("Logout", "unrendered")

    else:
        try:
            authenticator.experimental_guest_login("🔐 Login con Google",
                                                   provider="google",
                                                   oauth2=config['oauth2'],
                                                   single_session=False)
            with open('userschh_login_google.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Error en login: {e}")
    st.markdown("</div>", unsafe_allow_html=True)




st.markdown("""
<style>
/* Contenedor horizontal sin salto */
.botonera-horizontal {
    display: flex;
    flex-wrap: nowrap;
    justify-content: center;
    gap: 20px;
    overflow-x: auto; /* Opcional: muestra scroll si no cabe */
    padding: 10px 0;
}

.botonera-horizontal > div {
    flex: 1 1 auto;
    max-width: 220px; /* opcional, si quieres limitar el tamaño de cada botón */
}
</style>
""", unsafe_allow_html=True)





# --- Contenido principal centrado ---
# Bloque principal centrado correctamente
with st.container():
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto; text-align: center;">
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="titulo-central fade-in">Bienvenido</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="subtitulo-central fade-in">¿Listo para aprender en libertad?</p>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)


    # --- Input: una fila arriba, ocupando mismo ancho que los botones ---
    #De esta forma se evita que los nombres de los autores salgan en varias líneas
    #Este funciona
    #input_col1, input_main, input_col2 = st.columns([2.5, 4.8, 2.7])
    #input_col1, input_main, input_col2 = st.columns([0.5, 4, 0.5]) #usar
    #input_col1, input_main, input_col2 = st.columns([0.7, 3.6, 0.7])
    #input_col1, input_main, input_col2 = st.columns([2.3, 5.4, 2.3]) #sirve pero no
    #input_col1, input_main, input_col2 = st.columns([2.2, 6.2, 2.2])
    input_col1, input_main, input_col2 = st.columns([1.5, 7, 1.5])
    #input_col1, input_main, input_col2 = st.columns([1.5, 6.8, 1.5]) #ojo

    




    with input_main:
        pregunta = st.text_input(
            "Todo comienza con una pregunta...",
            key="question",
            label_visibility="collapsed",
            placeholder="Todo comienza con una pregunta..."
        )


# --- Botones: misma proporción ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
#cols = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 2.5], gap="small")
#cols = st.columns([0.5, 1, 1, 1, 1, 0.5], gap="small")  #usar
#cols = st.columns([0.7, 1, 1, 1, 1, 0.7], gap="small")
#cols = st.columns([2.3, 1.5, 1.5, 1.5, 1.5, 2.3], gap="small")  #sirve pero no
#cols = st.columns([2.2, 1.55, 1.55, 1.55, 1.55, 2.2], gap="small")
cols = st.columns([1.5, 1.75, 1.75, 1.75, 1.75, 1.5], gap="small")






# Inicializar mensaje principal si no existe
if "error_message_principal" not in st.session_state:
    st.session_state["error_message_principal"] = ""

def manejar_click_autor(nombre_autor, pagina_destino):
    if not pregunta.strip():
        st.session_state["error_message_principal"] = "✏️ Por favor, escribe una pregunta antes de continuar."
    elif not st.session_state.get("authentication_status"):
        st.session_state["error_message_principal"] = "🔒 Debes iniciar sesión para poder continuar."
    else:
        st.session_state["autor"] = nombre_autor
        nuevo_id = str(uuid.uuid4())
        usuario = st.session_state.get("username", "")
        mensaje_inicial = pregunta.strip()
        autor_key = nombre_autor
        mensaje_usuario = [{"role": "user", "content": mensaje_inicial}]

        # Guardar en sesión para que lo cargue el otro lado
        st.session_state[f"chat_id_{autor_key}"] = nuevo_id
        st.session_state[f"messages_{autor_key}"] = mensaje_usuario
        st.session_state["autor_a_redirigir"] = autor_key
        st.session_state["cargar_chat_especifico"] = True
        st.session_state["redirigir_forzado"] = True

        DynamoDatabase.save(
            nuevo_id,
            usuario,
            autor_key,
            "nuevo chat",
            mensaje_usuario
        )

        st.switch_page(pagina_destino)

with cols[1]:
    if st.button("📚 Frederich A. Hayek", key="botonazo_hayek"):
        manejar_click_autor("hayek", "pages/hayek.py")

with cols[2]:
    if st.button("💡 Henry Hazlitt"):
        manejar_click_autor("hazlitt", "pages/hazlitt.py")

with cols[3]:
    if st.button("🏛️ Ludwig Von Mises"):
        manejar_click_autor("mises", "pages/mises.py")

with cols[4]:
    if st.button("🌐 Todos los autores"):
        manejar_click_autor("general", "pages/todos_autores.py")

# --- Texto permanente informativo ---
st.markdown("""
<p style='text-align: center; max-width: 600px; margin: 30px auto 0 auto; font-size: 0.95rem;'>
    Con este chat aprenderás los <strong>principios éticos, jurídicos y económicos</strong>
    de una sociedad de personas libres y responsables.
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)


# --- Mensaje de error general centrado (si aplica) ---
if st.session_state["error_message_principal"]:
    st.markdown(f"""
        <div style='width: 100%; max-width: 900px; margin: 20px auto;'>
            <div style='background-color: #ffe6e6; color: #a80000;
                        padding: 15px 25px; border-radius: 10px;
                        border: 1px solid #f5c2c7; text-align: center;
                        font-size: 0.95rem;'>
                 {st.session_state["error_message_principal"]}
            </div>
        </div>
    """, unsafe_allow_html=True)