import streamlit as st
import config.dynamo_crud as DynamoDatabase
import uuid
from config.model_ia import run_hayek_chain
from config.sugerencias_preguntas import get_sugerencias_por_autor
from config.model_ia import extract_citations, parse_s3_uri
import streamlit.components.v1 as components
import streamlit_authenticator as stauth

def authenticated_menu():
    st.sidebar.success(f"Usuario: {st.session_state.username}")
    with st.sidebar:
        components.html("""
        <style>
            .btn-print {
                background-color: #ffffff;
                color: #262730;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: 0.45rem 1rem;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                width: 100%;
                transition: background-color 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            }

            .btn-print:hover {
                background-color: #f0f2f6;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
            }
        </style>

        <button class="btn-print" onclick="window.top.print()">🖨️ Print</button>
    """, height=50)
    st.sidebar.markdown("### Chatbots disponibles:")
    st.sidebar.page_link("hayek.py", label="Friedrich A. Hayek")
    st.sidebar.page_link("pages/hazlitt.py", label="Henry Hazlitt")
    st.sidebar.page_link("pages/mises.py", label="Ludwig von Mises")
    st.sidebar.page_link("pages/todos_autores.py", label="Todos los autores ")
    st.sidebar.markdown('<hr style="margin-top:4px; margin-bottom:4px;">', unsafe_allow_html=True)

def unauthenticated_menu():
    st.sidebar.page_link("hayek.py", label="Log in")

def main():
    session = st.session_state.username
    titulo = "Friedrich A. Hayek 🔗"
    author = "hayek"
    mensaje_nuevo_chat = "Nuevo chat con Friedrich A. Hayek"

    st.subheader(titulo, divider='rainbow')

    # Estado inicial separado por autor
    if "messages_hayek" not in st.session_state:
        st.session_state.messages_hayek = []
    if "chat_id_hayek" not in st.session_state:
        st.session_state.chat_id_hayek = ""
    if "new_chat_hayek" not in st.session_state:
        st.session_state.new_chat_hayek = False

    def cleanChat():
        st.session_state.new_chat_hayek = False

    def cleanMessages():
        st.session_state.messages_hayek = []

    def loadChat(chat, chat_id):
        st.session_state.new_chat_hayek = True
        st.session_state.messages_hayek = chat
        st.session_state.chat_id_hayek = chat_id

    with st.sidebar:
        st.title(titulo)

        if st.button(mensaje_nuevo_chat, icon=":material/add:", use_container_width=True):
            st.session_state.chat_id_hayek = str(uuid.uuid4())
            DynamoDatabase.save(st.session_state.chat_id_hayek, session, author, "nuevo chat", [])
            st.session_state.new_chat_hayek = True
            cleanMessages()
            st.session_state["hayek_suggested"] = get_sugerencias_por_autor("hayek")

        datos = DynamoDatabase.getChats(session, author)

        if datos:
            for item in datos:
                chat_id = item["SK"].split("#")[1]
                if f"edit_mode_{chat_id}" not in st.session_state:
                    st.session_state[f"edit_mode_{chat_id}"] = False

                with st.container():
                    c1, c2, c3 = st.columns([8, 1, 1])

                    c1.button(f"  {item['Name']}", type="tertiary", key=f"id_{chat_id}", on_click=loadChat,
                              args=(item["Chat"], chat_id), use_container_width=True)

                    c2.button("", icon=":material/edit:", key=f"edit_btn_{chat_id}", type="tertiary", use_container_width=True,
                              on_click=lambda cid=chat_id: st.session_state.update(
                                  {f"edit_mode_{cid}": not st.session_state[f"edit_mode_{cid}"]}))

                    #c3.button("", icon=":material/delete:", key=f"delete_{chat_id}", type="tertiary", use_container_width=True,
                    #          on_click=DynamoDatabase.delete, args=(chat_id, session, author))
                    
                    ##Nuevo boton de eliminacion

                    c3.button("",icon=":material/delete:",key=f"delete_{chat_id}",type="tertiary",use_container_width=True,
                            on_click=lambda cid=chat_id: (
                                DynamoDatabase.delete(cid, session, author),
                                st.session_state.update({
                                    "messages_hayek": [],
                                    "chat_id_hayek": "",
                                    "new_chat_hayek": False
                                }) if st.session_state.get("chat_id_hayek") == cid else None,
                            )
                            )
                    
                    if st.session_state[f"edit_mode_{chat_id}"]:
                        new_name = st.text_input("Nuevo nombre de chat:", value=item["Name"], key=f"rename_input_{chat_id}")
                        if st.button("✅ Guardar nombre", key=f"save_name_{chat_id}"):
                            DynamoDatabase.editNameManual(chat_id, new_name, session, author)
                            st.session_state[f"edit_mode_{chat_id}"] = False
                            st.rerun()

                st.markdown('<hr style="margin-top:4px; margin-bottom:4px;">', unsafe_allow_html=True)
        else:
            st.caption("No tienes conversaciones guardadas.")

    if st.session_state.new_chat_hayek:
        if st.session_state.get("hayek_suggested"):
            st.markdown("##### 💬 Sugerencias de preguntas")
            cols = st.columns(4)
            for i, question in enumerate(st.session_state["hayek_suggested"]):
                with cols[i]:
                    if st.button(question, key=f"suggestion_{i}"):
                        st.session_state["suggested_prompt_hayek"] = question
                        st.rerun()

        for message in st.session_state.messages_hayek:
            with st.chat_message(message["role"]):
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

        prompt = st.chat_input("Puedes escribir aquí...")
        if not prompt and "suggested_prompt_hayek" in st.session_state:
            prompt = st.session_state.pop("suggested_prompt_hayek")

        if prompt:
            st.session_state.messages_hayek.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                full_context = None

                for chunk in run_hayek_chain(prompt, st.session_state.messages_hayek):
                    if 'response' in chunk:
                        full_response += chunk['response']
                        placeholder.markdown(full_response)
                    elif 'context' in chunk:
                        full_context = chunk['context']
                placeholder.markdown(full_response)

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
                        s3_uri = citation["metadata"]["source"]
                        bucket, key = parse_s3_uri(s3_uri)
                        file_name = key.split("/")[-1].split(".")[0]
                        st.markdown(f"**📄 Fuente:** `{file_name}`")
                        st.markdown("---")

            st.session_state.messages_hayek.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations
            })

            DynamoDatabase.edit(
                st.session_state.chat_id_hayek,
                st.session_state.messages_hayek,
                session,
                author
            )

            if DynamoDatabase.getNameChat(st.session_state.chat_id_hayek, session, author) == "nuevo chat":
                DynamoDatabase.editName(st.session_state.chat_id_hayek, prompt, session, author)
                st.rerun()

    else:
        st.success("Puedes crear o seleccionar un chat existente")

def authenticator_login():
    st.set_page_config(
        page_title="Chatbot CHH",
        page_icon="📘",
    )
    import yaml
    from yaml.loader import SafeLoader
    with open('userschh.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    if "show_register_form" not in st.session_state:
        st.session_state["show_register_form"] = False

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    authenticator.login(single_session=False, fields={
        'Form name': 'Iniciar Sesión',
        'Username': 'Email',
        'Password': 'Contraseña',
        'Login': 'Iniciar sesión'
    })

    if st.session_state["authentication_status"]:
        authenticator.logout(button_name="Cerrar Sesión", location='sidebar')
        authenticated_menu()
        main()

    elif st.session_state["authentication_status"] is False:
        st.error('Nombre de usuario / Contraseña es incorrecta')
    elif st.session_state["authentication_status"] is None:
        st.warning('Por favor introduzca su nombre de usuario y contraseña')

    if not st.session_state["authentication_status"]:
        if st.button("Registrar nuevo usuario"):
            st.session_state["show_register_form"] = True

        if st.session_state["show_register_form"]:
            try:
                email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
                    merge_username_email=True,
                    captcha=False,
                    fields={
                        'Form name': 'Registrar usuario',
                        'First name': 'Nombre',
                        'Last name': 'Apellido',
                        'Email': 'Email',
                        'Password': 'Contraseña',
                        'Repeat password': 'Repetir contraseña',
                        'Password hint': 'Pista de contraseña (Ingresa una frase que te ayude a recordarla)',
                        'Register': 'Registrar Usuario'
                    }
                )
                if email_of_registered_user:
                    st.success('Usuario registrado exitosamente, por favor inicia sesión con tu correo y contraseña')
                    st.session_state["show_register_form"] = False
            except Exception as e:
                st.error(e)

            with open('userschh.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    authenticator_login()