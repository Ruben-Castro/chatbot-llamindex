import os
import streamlit as st
import time 

from dotenv import load
from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode


load()


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print("RAG....")

    index_name = "llama-index-documentation-helper"
    pincone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pincone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )


index = get_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True
    )


st.set_page_config(
    page_title="Chat with llama index docs powered by llama index",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex docs")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role":"assistant",
            "content": "Ask me a question about LlamaIndex's open source python library"
        },
    ]

if prompt:= st.chat_input("Your Question"):
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })


for message in st.session_state.messages:

    with st.chat_message(message['role']):
        st.write(message['content'])



if st.session_state.messages[-1]['role'] != "assistant":
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node:{i+1} score={node.score}")
                    st.write(node.text)

          
            message = {
                'role':"assistant",
                'content': response.response
            }

            st.session_state.messages.append(message)