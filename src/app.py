import streamlit as st
from Rag import launch_depression_assistant, depression_assistant
# from openai import OpenAI
from together import Together
from dotenv import load_dotenv

# --- Sidebar ---
st.sidebar.title("Settings")

# you can add the embedder model to tried out more
# but it has to be from sentence-transformers library,
# if not, need to adapt the code to load the embedder model
model_options = [
    "all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v3",
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
# Show title and description.
st.title("💬 Depression Assistant Chatbot")


# API provider selection
api_provider = st.sidebar.selectbox(
    "Select API Provider",
    ["Default Free Together AI API","OpenAI", "Together AI", "NVIDIA"]
)

# Only show API key input if not using default free API
if api_provider != "Default Free Together AI API":
    # Dynamic API key input based on selected provider
    api_key = st.text_input(f"{api_provider} API Key", type="password")
    if not api_key:
        st.info(f"Please add your {api_provider} API key to continue.", icon="🗝️")
else:
    try:
        api_key = st.secrets["TOGETHER_API_KEY"]  # Replace with actual default key
        llm_client =Together(api_key=api_key)
    except Exception as e:
        st.warning("Default API key not found. Probably running locally")
        llm_client = None

openai_api_key = api_key if api_provider == "OpenAI" else None
together_api_key = api_key if api_provider == "Together AI" else None
nvidia_api_key = api_key if api_provider == "NVIDIA" else None

# Initialize LLM client based on provider
if api_provider == "OpenAI":
    if openai_api_key:
        st.warning("Open AI entry is not yet tested in this app. Because we haven't implement the function to select model. Please use Together AI instead.")
        llm_client = OpenAI(api_key=openai_api_key)
elif api_provider == "Together AI":
    llm_client = Together(together_api_key)
elif api_provider == "NVIDIA":
    st.warning("NVIDIA entry is not yet tested in this app. Please use Together AI instead.")
    # llm_client = NvidiaLLM(nvidia_api_key)
    llm_client = None
else:
    st.warning("No valid API key provided. Using default Free Together AI API.")
    llm_client = None

        
    # --- Track launch state and embedder ---
    if "launched" not in st.session_state:
        st.session_state.launched = False
    if "assistant_launched" not in st.session_state:
        st.session_state.assistant_launched = False

    # --- Launch button ---
    if not st.session_state.launched:
        st.write("This is a simple depression assistant bot. You can ask questions related to depression and get responses based on clinical guidelines.")
        st.write(
        "This is a simple chatbot that uses **RAG (Retrieval-Augmented Generation)** to answer questions related to depression. "
        "It uses a combination of a retrieval model to find relevant documents and a language model to generate responses based on those documents."
        )
        st.write("You can choose different embedder models from the sidebar to see how they affect the responses.")
        st.write("You can also choose different API providers from the side bar to use different models and see how they affect the responses.")
        st.write("Please click the button below to launch the assistant. You'll have to reload the page to change the embedder model or API provider.")

        if st.button("Launch Assistant"):
            st.session_state.launched = True
            st.rerun()

    # --- After launch ---
    if st.session_state.launched:
        if not st.session_state.assistant_launched:
            launch_depression_assistant(embedder_name=embedder_name,designated_client=llm_client)
            st.session_state.assistant_launched = True

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        

        # React to user input
        if prompt := st.chat_input("Ask me questions about the CANMAT depression guideline!"):
            # results, response = depression_assistant(prompt)
            # Divide the web page into two parts: left for results, right for response
            
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            placeholder = st.chat_message("assistant").empty()
            collected_text = ""
            results, response = depression_assistant(prompt, True)

            for chunk in response:
                collected_text += chunk
                placeholder.markdown(collected_text)

       
            st.session_state.messages.append({"role": "assistant", "content": collected_text})

            # st.chat_message("assistant").markdown(response)
            # st.session_state.messages.append({"role": "assistant", "content": response})
