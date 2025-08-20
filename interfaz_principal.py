import streamlit as st
st.set_page_config(page_title="Interfaz Principal", layout="wide")
st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)

from config.dynamo_crud import getChats
import uuid
from config import dynamo_crud as DynamoDatabase


import os
# Healthcheck endpoint simulado
if st.query_params.get("check") == "1":
    st.markdown("OK")
    st.stop()

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
        "label": "Manuel F. Ayau (Muso)",
        "key": "muso",
        "logo": "",
        "avatar_size": 20,
        "pagina": "muso"
    },
    {
        "label": "Todos los autores",
        "key": "general",
        "logo": "",
        "avatar_size": 20,
        "pagina": "todos_autores"
    }

]


# ---------- CSS general ----------
st.markdown("""
<style>
            
/*  Eliminar espacio superior predeterminado en Streamlit */
    header[data-testid="stHeader"] {
        height: 0rem !important;
        padding: 0 !important;
        margin: 0 !important;
        background-color: transparent !important;
    }

    div.block-container {
        padding-top: 0rem !important;
    }
            
    .contenedor-grid {
        max-width: 1140px;
        margin: 0 auto;
        padding: 0 15px;
    }

    .titulo-central {
        text-align: center;
        margin-top: 0;
        margin-bottom: 25px;
        font-size: 28px;
        font-weight: 600;
    }
            
    .subtitulo-central {
        margin-top: 0;
        text-align: center;
        font-size: 18px;
    }

    .fade-in {
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
         

    /*  Card principal */
    div[st-key="card_wrap"] {
        background: rgba(0, 255, 0, 0.05);
        border: 1.5px solid #d6081f;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
        margin-top: 10px;
    }

    .texto-final {
        margin-top: 25px;
        font-size: 18px;
        text-align: center;
    }
    
/*  Contenedor principal del botón Login */
.stLinkButton {
    display: flex !important;
    justify-content: center;
    align-items: center;
    padding: 0 !important;
    margin-top: 6px;
}

/*  Botón de login  */
a[data-testid="stBaseLinkButton-secondary"] {
    border-radius: 8px !important;  /* igual que historial */
    border: 1.5px solid #d6081f !important;
    background-color: white !important;
    color: black !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    text-decoration: none !important;
    transition: all 0.3s ease;
}

/*  Hover Login */
a[data-testid="stBaseLinkButton-secondary"]:hover {
    background-color: #d6081f !important;
    color: white !important;
}

/*----------------------Boton historial-------------------*/

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
            
div[class*="st-key-btn_historial"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;  

}

div[class*="st-key-btn_historial"] div[data-testid="stTooltipHoverTarget"] {
    justify-content: center !important;
    width: 100% !important;
}            

            
            
/*------------------Boton Logout--------------*/
            
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

                        
/*------------------Boton Login--------------*/
            
div[class*="st-key-btn_propio_login"] > div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
}
         
            
div[class*="st-key-btn_propio_login"] button {
    /*border-radius: 25px;*/
    border-radius: 8px !important;  /* Más cuadrado */
    border: 1.5px solid #d6081f !important;
    background-color: white !important;
    color: black !important;
    font-size: 16px;
    padding: 6px 14px;
    transition: all 0.3s ease;
    display: flex !important;
    justify-content: flex-end !important;  /* botón pegado a la derecha */
    align-items: center !important;
}
    
            
   
                    
div[class*="st-key-btn_propio_login"] button:hover {
    background-color: #d6081f !important;
    color: white !important;
}                
            

/*------------------Modal de historial-----------*/           
/* Ampliar el diálogo */
div[data-testid="stDialog"] div[role="dialog"] {
    width: 90vw !important;
    max-width: 550px !important; 
}
            

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
            
/* Estilo solo para los botones 'Abrir conversación' */
div[data-testid="stDialog"] [class*="st-key-open_"] button {
    border: none !important;
    background: none !important;
    color: #d6081f !important; /* ← Aquí le das color al ícono */
    font-size: 18px;
    padding: 4px;
    box-shadow: none !important;
}

    /* Cambia color en hover */
    div[data-testid="stDialog"] [class*="st-key-open_"] button:hover {
        background-color: rgba(0,0,0,0.05);
        border-radius: 50%;
    }
            

/**Input pregunta inicial**/
                        
div.st-key-first_question .stTextInput input {
    border: 1.5px solid #d60812 !important;
    border-radius: 10px;
    padding: 0.5rem !important;
    background-color: white !important;
    font-size: 18px !important;
}
            

div.st-key-enviar_hayek div.stButton > button,
div.st-key-enviar_hazlitt div.stButton > button,
div.st-key-enviar_mises div.stButton > button,
div.st-key-enviar_general div.stButton > button,
div.st-key-enviar_muso div.stButton > button
   {
    border-radius: 25px;
    padding: 0.5rem 1.5rem;
    border: 1.5px solid #d60812 !important;
    color: black !important;
    background-color: white !important;
    transition: all 0.3s ease;
    font-weight: 500;
    font-size: 50px !important;
}

div.st-key-enviar_hayek div.stButton > button:hover,
div.st-key-enviar_hazlitt div.stButton > button:hover,
div.st-key-enviar_mises div.stButton > button:hover,
div.st-key-enviar_general div.stButton > button:hover,
div.st-key-enviar_muso div.stButton > button:hover {
    background-color: #d60812 !important;
    color: white !important;
}


                   
div.st-key-enviar_hayek button div p,
div.st-key-enviar_hazlitt button div p,
div.st-key-enviar_mises button div p,
div.st-key-enviar_general button div p , 
div.st-key-enviar_muso button div p{
    font-size: 17px !important;
}
            


           
</style>
<div class='contenedor-grid'>
""", unsafe_allow_html=True)

mostrar_columnas = False #Para mostrar colores en la pantalla.
mostrar_bordes= False #Mostrar bordes de las cols


#Dialog a mostrar
@st.dialog("🕘 Historial de conversaciones")
def mostrar_historial():
    usuario = st.user.email ## st.session_state.get("username", "")
    
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
                    #st.markdown(f"""
                    #    <div style="display: flex; align-items: center; gap: 8px;">
                    #        <span style="font-weight: 600; font-size: 18px;">{autor['label']}</span>
                    #        <span style="font-size: 20px;">🧾</span>
                    #    </div>
                    #""", unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-weight: 600; font-size: 18px;">{autor['label']}</span>
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

                #c1, c2 = st.columns([7.5, 0.5]) #No recomendable en streamlit, el uso de proporciones no enteras, tienden a deformarse en pantallas pequeñas

                c1, c2 = st.columns([15, 1])

                if mostrar_columnas:  
                    with c1:
                        st.markdown(f"""
                        <div style='
                            background-color: rgba(100, 200, 255, 0.15);
                            border: 1px dashed rgba(50,50,50,0.4);
                            padding: 6px;
                            text-align: center;
                            font-size: 12px;
                            color: #333;
                            margin-bottom: 6px;
                        '>
                                C1
                        </div>
                        """, unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"""
                            <div style='
                                background-color: rgba(255, 180, 100, 0.15);
                                border: 1px dashed rgba(50,50,50,0.4);
                                padding: 6px;
                                text-align: center;
                                font-size: 12px;
                                color: #333;
                                margin-bottom: 6px;
                            '>
                                C2
                            </div>
                            """, unsafe_allow_html=True)

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
            if autor:  # Solo redirigir 
                    if autor == "general":
                        st.switch_page("pages/todos_autores.py") 
                    else:
                        st.switch_page(f"pages/{autor}.py")


#cols_top = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])


# ---------- Header:  ----------
cols_top = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])

# Visualizador de columnas
for i, col in enumerate(cols_top):
    with col:
        if mostrar_columnas: 
            st.markdown(f"""
                <div style='
                    background-color: rgba({(i*25)%255}, {(i*50)%255}, {(i*75)%255}, 0.1);
                    border: 1px dashed rgba(100,100,100,0.4);
                    padding: 6px;
                    text-align: center;
                    font-size: 12px;
                    color: #555;
                    margin-bottom: 6px;
                '>
                    Col {i}
                </div>
            """, unsafe_allow_html=True)

with cols_top[0]:
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://intranet.ufm.edu/reportesai/img_chatbot/LOGO_UFM_FullCol.png' width='150'>
        </div>
    """, unsafe_allow_html=True)

with cols_top[10]:


    if st.user.is_logged_in:

        #username = st.user.email #st.session_state.get("username")
        profile_pic_url = st.user.picture
        correo =st.user.email
        #col1, col2, col3 = st.columns([1, 1, 1])
        col1, col2, col3 = st.columns(3) # ,border=mostrar_bordes


        # Visualizador para las subcolumnas de la Col 10
        if mostrar_columnas:
            for i, col in enumerate([col1, col2, col3]):
                with col:
                    st.markdown(f"""
                        <div style='
                            background-color: rgba({(i*80)%255}, {(i*120)%255}, {(i*40)%255}, 0.1);
                            border: 1px dashed #d6081f;
                            padding: 6px;
                            text-align: center;
                            font-size: 12px;
                            color: #333;
                            margin-bottom: 6px;
                        '>
                            Sub-Col {i}
                        </div>
                    """, unsafe_allow_html=True)





        with col1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if profile_pic_url:
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; align-items: center;'>
                        <img src="{profile_pic_url}" alt="Usuario"
                             title="{correo}"
                             onerror="this.src='https://www.gravatar.com/avatar/?d=mp&f=y';"
                             style="width: 36px; height: 36px; object-fit: cover; border-radius: 50%; margin-top: 4px;" />
                    </div>  </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button("", icon=":material/description:", key="btn_historial", type="tertiary", help="Historial de conversaciones"):
                mostrar_historial()  # mostrar el historial

        with col3:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

            if st.button("", icon=":material/logout:", key="btn_propio_logout", help="Cerrar sesión"):
                st.logout()
            
            #if logout_button:
                #authenticator.logout("Logout", "unrendered")

    else:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("🔐 Login con Google", key="btn_propio_login"):
            st.login("google")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Título ----------

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div class='titulo-central fade-in'> <h2> Bienvenido, ¿listo para aprender en libertad? </h2> </div>", unsafe_allow_html=True )
st.markdown("""<div class="subtitulo-central">✍️ <strong>Escribe una pregunta</strong> y luego selecciona al autor con quien deseas conversar.</div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------- Tarjeta central con columnas visualizadas ----------

col_left, col_card, col_right = st.columns([3, 10, 3],border=mostrar_bordes)

#col_left, col_card, col_right = st.columns([2, 9, 2],border=mostrar_bordes)


# Visualizador de estructura 3 columnas
for i, col in enumerate([col_left, col_card, col_right]):
    with col:
        if mostrar_columnas: 

            st.markdown(f"""
                <div style='
                    background-color: rgba({(i*100)%255}, {(i*150)%255}, {(i*50)%255}, 0.1);
                    border: 1px dashed #888;
                    padding: 6px;
                    text-align: center;
                    font-size: 12px;
                    color: #555;
                    margin-bottom: 6px;
                '>
                    Card Col {i}
                </div>
            """, unsafe_allow_html=True)



# ---------- Card central contenido ----------
with col_card:
    with st.container(key="card_wrap",border=mostrar_bordes):
        #st.markdown("<br>", unsafe_allow_html=True)


        
        pregunta = st.text_input(
            label=" ",
            key="first_question",
            label_visibility="visible",
            placeholder="Todo comienza con una pregunta...",
            help="✍️ Escribe una pregunta y luego selecciona al autor con quien deseas conversar."
        )        


        #Para redirigir autores:
        # Inicializar mensaje principal si no existe
        if "error_message_principal" not in st.session_state:
            st.session_state["error_message_principal"] = ""

        def manejar_click_autor(nombre_autor, pagina_destino):
            if not pregunta.strip():
                st.session_state["error_message_principal"] = "✍️ Por favor, escribe tu pregunta antes de seleccionar un autor."
            elif not st.user.is_logged_in:
                st.session_state["error_message_principal"] = "🔒 Debes iniciar sesión para poder continuar."
            else:
                st.session_state["autor"] = nombre_autor
                nuevo_id = str(uuid.uuid4())
                usuario = st.user.email  ##st.session_state.get("username", "")
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

        #st.markdown("<br>", unsafe_allow_html=True)
        # Botones autores
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        #cols_autores = st.columns(4,gap="small",border=mostrar_bordes)
        cols_autores = st.columns(5,gap="small",border=mostrar_bordes)


        for i, col in enumerate(cols_autores):
            with col:
                if mostrar_columnas: 
                    # Bloque visual para debug 
                    st.markdown(f"""
                        <div style='
                            background-color: rgba({(i*100)%255}, {(i*150)%255}, {(i*50)%255}, 0.1);
                            padding: 12px;
                            border: 1px dashed rgba(0,0,0,0.2);
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-size: 13px;
                            color: #333;
                        '>
                            Columna {i+1}
                        </div>
                    """, unsafe_allow_html=True)

     
        with cols_autores[0]:
            if st.button("📚 Friedrich A. Hayek", key="enviar_hayek", use_container_width=True):
                manejar_click_autor("hayek", "pages/hayek.py")
        with cols_autores[1]:
            if st.button("💡 Henry Hazlitt", key="enviar_hazlitt", use_container_width=True):
                manejar_click_autor("hazlitt", "pages/hazlitt.py")
        with cols_autores[2]:
            if st.button("🏛️ Ludwig Von Mises", key="enviar_mises", use_container_width=True):
                manejar_click_autor("mises", "pages/mises.py")
        with cols_autores[3]:
            if st.button("🧠 Manuel F. Ayau (Muso)", key="enviar_muso", use_container_width=True):
                manejar_click_autor("muso", "pages/muso.py")
        with cols_autores[4]:
            if st.button("🌐 Todos los autores", key="enviar_general", use_container_width=True):
                manejar_click_autor("general", "pages/todos_autores.py")


        st.markdown("<br>", unsafe_allow_html=True)
        # Texto informativo
        st.markdown("""
        <div class='texto-final'>
            Con este chat aprenderás los <strong>principios éticos, jurídicos y económicos</strong><br>
            de una sociedad de personas libres y responsables.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # --- Mensaje de error general centrado (si aplica) ---
        if st.session_state["error_message_principal"]:
            st.markdown(f"""
                <div style='width: 100%; max-width: 900px; margin: 20px auto;'>
                    <div style='background-color: #ffe6e6; color: #a80000;
                                padding: 15px 25px; border-radius: 10px;
                                border: 1px solid #f5c2c7; text-align: center;
                                font-size: 17px;'>
                        {st.session_state["error_message_principal"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ---------- Cierre del grid ----------
st.markdown("</div>", unsafe_allow_html=True)
