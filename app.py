import streamlit as st
from Chatbot.RagChatbot import ask_hospital


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🏥 Hospital Analytics Assistant")

st.write(
    "Ask questions about hospital information or patient data."
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# QUESTION BOX
# =========================================================

question = st.chat_input("Ask your question...")


# =========================================================
# WHEN USER ASKS A QUESTION
# =========================================================

if question:

    # Add user's question to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user's question
    with st.chat_message("user"):
        st.write(question)


    # Get chatbot answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask_hospital(question)

        st.write(answer)


    # Add answer to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )