import streamlit as st
import uuid
from PIL import Image
from io import BytesIO
import base64
from config.model_ia import run_hazlitt_chain, extract_citations, parse_s3_uri
import config.dynamo_crud as DynamoDatabase
from config.sugerencias_preguntas import get_sugerencias_por_autor
import botocore.exceptions
import streamlit as st
import uuid
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
from collections import defaultdict
from langchain.schema import Document  
from dotenv import load_dotenv
from langsmith import traceable
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree
from langchain.callbacks import collect_runs


# Cargar variables de entorno
load_dotenv()


from PIL import Image
import base64
from io import BytesIO

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


st.set_page_config(page_title="Henry Hazlitt",layout="wide")



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




def image_to_base64(path):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

logo_assistant = "img/Henry-Hazlitt_32_64.png"

logo_ufm= "img/UFM-LOGO-MATOR.png"

#Henry_Hazlitt_24_48

logo_base64 =  image_to_base64("img/Henry_Hazlitt_48_96.png")
#image_to_base64("img/Henry_Hazlitt_24_48.png")
logo_url = "https://intranet.ufm.edu/reportesai/img_chatbot/Henry-Hazlitt-noblank.png"  # reemplaza con la URL real


def authenticated_menu():

    if st.sidebar.button("", icon=":material/arrow_back:", help="Regresar a menú principal", type="tertiary", key="button_back"):
        st.switch_page("interfaz_principal.py")

    # st.sidebar.success(f"Usuario: {st.session_state.username}")
      # Logo UFM más pequeño y centrado en sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px; margin-top: -10px;">
        <img src="https://intranet.ufm.edu/reportesai/img_chatbot/UFM-LOGO-MATOR.png" 
             style="width: 100%; max-width: 180px;">
    </div>
        <hr style='border: none; height: 2px; background-color: #d6081f; margin: 8px 0 16px 0;'>
""", unsafe_allow_html=True)
    
        

    st.sidebar.markdown("### Autores disponibles:")
    st.sidebar.page_link("pages/hayek.py", label="Friedrich A. Hayek")
    st.sidebar.page_link("pages/hazlitt.py", label="Henry Hazlitt")
    st.sidebar.page_link("pages/mises.py", label="Ludwig von Mises")
    st.sidebar.page_link("pages/todos_autores.py", label="Todos los autores ")
    st.sidebar.markdown("""
        <hr style='border: none; height: 2px; background-color: #d6081f; margin: 8px 0 16px 0;'>
        """, unsafe_allow_html=True)
    
    
def invoke_with_retries_hazlitt(run_chain_fn, question, history, config=None, max_retries=10, author= "hazlitt"):
    attempt = 0
    warning_placeholder = st.empty()
    
    # Esto es lo nuevo: usamos el bloque de mensaje del asistente UNA SOLA VEZ
    with st.chat_message("assistant",  avatar=logo_assistant):
        response_placeholder = st.empty()

        while attempt < max_retries:
            try:
                print(f"Reintento {attempt + 1} de {max_retries}, , chat_id_hazlitt {st.session_state.chat_id_hazlitt}")
                full_response = ""
                full_context = None

                for chunk in run_chain_fn(question, history):
                    if 'response' in chunk:
                        full_response += chunk['response']
                        response_placeholder.markdown(full_response)
                    elif 'context' in chunk:
                        full_context = chunk['context']

                response_placeholder.markdown(full_response)

                citations = []
                if full_context:
                    citations_objs = extract_citations(full_context)
                    citations = [{
                        "page_content": c.page_content,
                        "metadata": {
                            "source": c.metadata["location"]["s3Location"]["uri"],
                            "score": str(c.metadata.get("score", ""))
                        }
                    } for c in citations_objs]

                    with st.expander("📚 Referencias utilizadas en esta respuesta"):
                        for citation in citations:
                            st.markdown(f"**📝 Contenido:** {citation['page_content']}")
                            bucket, key = parse_s3_uri(citation["metadata"]["source"])
                            file_name = key.split("/")[-1].split(".")[0]
                            st.markdown(f"**📄 Fuente:** `{file_name}`")
                            st.markdown("---")

                st.session_state.messages_hazlitt.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations
                })

                DynamoDatabase.edit(
                    st.session_state.chat_id_hazlitt,
                    st.session_state.messages_hazlitt,
                    st.session_state.username,
                    author
                )

                if DynamoDatabase.getNameChat(st.session_state.chat_id_hazlitt, st.session_state.username, author) == "nuevo chat":
                    DynamoDatabase.editName(st.session_state.chat_id_hazlitt, question, st.session_state.username, author)
                    st.rerun()

                warning_placeholder.empty()
                #print(f"Error inesperado en reintento {attempt}: {str(e)}")

                return

            except Exception as e:
                attempt += 1
                if attempt == 1:
                    warning_placeholder.markdown("⌛ Esperando generación de respuesta...", unsafe_allow_html=True)
                print(f"Error inesperado en reintento {attempt}: {str(e)}")
                if attempt == max_retries:
                    warning_placeholder.markdown("⚠️ **No fue posible generar la respuesta, vuelve a intentar.**", unsafe_allow_html=True)

def main():
    session = st.session_state.username
    titulo = "Henry Hazlitt"
    author = "hazlitt"
    mensaje_nuevo_chat = "Nueva conversación con Henry Hazlitt"

    


        # Si se solicitó cargar una conversación específica desde otro lugar
    if st.session_state.get("autor_a_redirigir") == "hazlitt":
        if st.session_state.get("cargar_chat_especifico"):
            chat_id = st.session_state.get("chat_id_hazlitt")
            datos_chat = DynamoDatabase.getChats(session, "hazlitt")
            chat_encontrado = next((item for item in datos_chat if item["SK"].endswith(chat_id)), None)
            if chat_encontrado:
                st.session_state.messages_hazlitt = chat_encontrado["Chat"]
                st.session_state.new_chat_hazlitt = True
            st.session_state.cargar_chat_especifico = False
        else:
            # Ir al autor sin cargar ningún chat
            st.session_state.chat_id_hazlitt = ""
            st.session_state.messages_hazlitt = []
            st.session_state.new_chat_hazlitt = False

        # Limpiar banderas
        st.session_state["autor_a_redirigir"] = ""
        st.session_state["cargar_chat_especifico"] = False




    st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 0;">
 <h3 style="margin: 0;">{titulo}     <img src="{logo_url}"  height="61" style="margin-left: 6px;">
</h3>
</div>
<hr style='border: 2px solid #d6081f; margin-top: 0; margin-bottom: 24px;'>
""", unsafe_allow_html=True)

    # Estado inicial separado por autor
    if "messages_hazlitt" not in st.session_state:
        st.session_state.messages_hazlitt = []
    if "chat_id_hazlitt" not in st.session_state:
        st.session_state.chat_id_hazlitt = ""
    if "new_chat_hazlitt" not in st.session_state:
        st.session_state.new_chat_hazlitt = False

    def cleanChat():
        st.session_state.new_chat_hazlitt = False

    def cleanMessages():
        st.session_state.messages_hazlitt = []

    def loadChat(chat, chat_id):
        st.session_state.new_chat_hazlitt = True
        st.session_state.messages_hazlitt = chat
        st.session_state.chat_id_hazlitt = chat_id

    with st.sidebar:
        st.title(titulo)

        if st.button(mensaje_nuevo_chat, icon=":material/add:", use_container_width=True):
            st.session_state.chat_id_hazlitt = str(uuid.uuid4())
            DynamoDatabase.save(st.session_state.chat_id_hazlitt, session, author, "nuevo chat", [])
            st.session_state.new_chat_hazlitt = True
            cleanMessages()
            st.session_state["hazlitt_suggested"] = get_sugerencias_por_autor("hazlitt")

        datos = DynamoDatabase.getChats(session, author)

        if datos:
            for item in datos:
                chat_id = item["SK"].split("#")[1]
                if f"edit_mode_{chat_id}" not in st.session_state:
                    st.session_state[f"edit_mode_{chat_id}"] = False

                with st.container():
                    c1, c2, c3 = st.columns([8, 1, 1])
                    c1.button(f"  {item['Name']}",
                            type="tertiary",
                            key=f"id_{chat_id}",
                            on_click=loadChat,
                            args=(item["Chat"], chat_id),
                            use_container_width=True)

                    c2.button("", icon=":material/edit:", key=f"edit_btn_{chat_id}", type="tertiary", use_container_width=True,
                              on_click=lambda cid=chat_id: st.session_state.update(
                                  {f"edit_mode_{cid}": not st.session_state[f"edit_mode_{cid}"]}))

                    c3.button("", icon=":material/delete:", key=f"delete_{chat_id}", type="tertiary", use_container_width=True,
                              on_click=lambda cid=chat_id: (
                                  DynamoDatabase.delete(cid, session, author),
                                  st.session_state.update({
                                      "messages_hazlitt": [],
                                      "chat_id_hazlitt": "",
                                      "new_chat_hazlitt": False
                                  }) if st.session_state.get("chat_id_hazlitt") == cid else None,
                              ))

                    if st.session_state[f"edit_mode_{chat_id}"]:
                        new_name = st.text_input("Nuevo nombre de chat:", value=item["Name"], key=f"rename_input_{chat_id}")
                        if st.button("✅ Guardar nombre", key=f"save_name_{chat_id}"):
                            DynamoDatabase.editNameManual(chat_id, new_name, session, author)
                            st.session_state[f"edit_mode_{chat_id}"] = False
                            st.rerun()

                st.markdown(""" <hr style='border: none; height: 1px; background-color: #d6081f; margin: 8px 0 16px 0;'> """, unsafe_allow_html=True)
                
        else:
            st.caption("No tienes conversaciones guardadas.")

  

    if st.session_state.new_chat_hazlitt:
        # Cargar sugerencias si no vienen desde pantalla principal
        if st.session_state.get("new_chat_hazlitt"):
            if "hazlitt_suggested" not in st.session_state:
                st.session_state["hazlitt_suggested"] = get_sugerencias_por_autor("hazlitt")

            st.markdown("##### 💬 Sugerencias de preguntas")
            cols = st.columns(4)
            for i, question in enumerate(st.session_state["hazlitt_suggested"]):
                with cols[i]:
                    if st.button(question, key=f"suggestion_{i}"):
                        st.session_state["suggested_prompt_hazlitt"] = question
                        st.rerun()

        for message in st.session_state.messages_hazlitt:
            avatar = logo_assistant if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "citations" in message:
                    with st.expander("📚 Referencias utilizadas en esta respuesta"):
                        for citation in message["citations"]:
                            st.markdown(f"**📝 Contenido:** {citation['page_content']}")
                            s3_uri = citation["metadata"]["source"]
                            bucket, key = parse_s3_uri(s3_uri)
                            file_name = key.split("/")[-1].split(".")[0]
                            st.markdown(f"**📄 Fuente:** `{file_name}`")
                            st.markdown("---")

        # --- Verificar si se debe invocar automáticamente el LLM ---
        ultimo_mensaje = st.session_state.messages_hazlitt[-1] if st.session_state.messages_hazlitt else None
        es_ultimo_user = ultimo_mensaje and ultimo_mensaje["role"] == "user"
        ya_hay_respuesta = any(m["role"] == "assistant" for m in st.session_state.messages_hazlitt)

        nombre_chat_actual = DynamoDatabase.getNameChat(
            st.session_state.chat_id_hazlitt,
            st.session_state.username,
            "hazlitt"
        )

        if es_ultimo_user and not ya_hay_respuesta and nombre_chat_actual == "nuevo chat":
            pregunta_inicial = ultimo_mensaje["content"]

            invoke_with_retries_hazlitt(
                run_chain_fn=run_hazlitt_chain,
                question=pregunta_inicial,
                history=[],
                author="hazlitt"
            )


        prompt = st.chat_input("Todo comienza con una pregunta...",key="chat_input_hazlitt")

        st.markdown("""
    <style>
    div.st-key-chat_input_hazlitt::after {
        content: "Este asistente puede cometer errores.";
        display: block;
        text-align: center;
        font-size: 0.75rem;
        color: #999;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)
        
        if not prompt and "suggested_prompt_hazlitt" in st.session_state:
            prompt = st.session_state.pop("suggested_prompt_hazlitt")

        if prompt:
            st.session_state.messages_hazlitt.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            invoke_with_retries_hazlitt(run_hazlitt_chain, prompt, st.session_state.messages_hazlitt)
    else:
        st.success("Puedes crear o seleccionar un chat existente")


def authenticator_login():


    with open('userschh_login_google.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )


    authenticator.login(fields=None, max_concurrent_users=None)


    # Si no hay sesión activa, detener
    if not st.session_state.get("authentication_status"):
        st.error("❗ Tu sesión ha expirado o no está activa. Vuelve a la pantalla principal.")
        st.switch_page("interfaz_principal.py") 
        st.stop()

    # Si todo bien, aseguramos que el username esté presente
    st.session_state["username"] = st.session_state.get("username")


        # --- Layout: columna vacía + avatar + logout alineado a la derecha ---
    espacio, avatar_col, logout_col = st.columns([8.8, 1.0, 1.2])

    st.markdown("""
    <style>
    /* Estilo base */
    div[class*="st-key-btn_propio_logout"] button {
        border: 1.5px solid #d6081f !important;
        background-color: white !important;
        color: black !important;
        font-size: 16px;
        padding: 6px 14px;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }

    /* Solo en pantallas grandes, acercarlo más al avatar */
    @media (min-width: 1600px) {
        div[class*="st-key-btn_propio_logout"] {
            margin-left: -50px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


    user_data = config['credentials']['usernames'].get(st.session_state["username"], {})
    profile_pic_url = user_data.get("picture", "")
    correo = user_data.get("email", "Correo no disponible")

    with avatar_col:
        if profile_pic_url:
            st.markdown(f"""
                <img src="{profile_pic_url}" alt=""
                    title="{correo}"
                    style="width: 36px; height: 36px; object-fit: cover; border-radius: 50%; margin-top: 4px;" />
            """, unsafe_allow_html=True)

    with logout_col:
        logout_button = st.button("", icon=":material/logout:", type="tertiary", key="btn_propio_logout",help="Cerrar sesión")
        if logout_button:
            authenticator.logout("Logout", "unrendered")
            st.rerun()
            st.markdown("""
            <script>
                document.cookie = "id_usuario=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
            </script>
            """, unsafe_allow_html=True)

            st.switch_page("interfaz_principal.py") 
    



    if st.session_state["authentication_status"]:
        authenticated_menu()
        main()

    elif st.session_state["authentication_status"] is False:
        st.error('Nombre de usuario / Contraseña es incorrecta')
    elif st.session_state["authentication_status"] is None:
        st.warning('Por favor introduzca su nombre de usuario y contraseña')


if __name__ == "__main__":
    authenticator_login()



    


