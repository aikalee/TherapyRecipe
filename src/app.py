import streamlit as st
from Rag import launch_depression_assistant, depression_assistant, streaming_depression_assistant

# --- Sidebar ---
st.sidebar.title("Settings")

# you can add the embedder model to tried out more
# but it has to be from sentence-transformers library,
# if not, need to adapt the code to load the embedder model
model_options = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "abhinand/MedEmbed-large-v0.1"
]
embedder_name = st.sidebar.selectbox(
    "Select embedder model",
    model_options,
    index=0
)

# --- Track launch state and embedder ---
if "launched" not in st.session_state:
    st.session_state.launched = False
if "assistant_launched" not in st.session_state:
    st.session_state.assistant_launched = False

# --- Launch button ---
if not st.session_state.launched:
    st.title("Depression Assistant")
    st.write("This is a simple depression assistant bot. You can ask questions related to depression and get responses based on clinical guidelines.")
    st.write("The embedder model is chosen from the sidebar.")

    if st.button("Launch Assistant"):
        st.session_state.launched = True
        st.rerun()

# --- After launch ---
if st.session_state.launched:
    if not st.session_state.assistant_launched:
        launch_depression_assistant(embedder_name=embedder_name)
        st.session_state.assistant_launched = True

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # results, response = depression_assistant(prompt)
        
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # st.chat_message("assistant").markdown(response)
        # st.session_state.messages.append({"role": "assistant", "content": response})

        placeholder = st.chat_message("assistant").empty()
        collected_text = ""

        for chunk in streaming_depression_assistant(prompt):
            collected_text += chunk
            placeholder.markdown(collected_text)

       
        st.session_state.messages.append({"role": "assistant", "content": collected_text})
