import streamlit as st
from rag import (
    setup_environment,
    load_rag_pipeline,
)

def streamlit_generate_response(rag_chain, input_text, session_id="default"):
    answer = ""
    placeholder = st.empty()
    for chunk in rag_chain.stream(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    ):
        try:
            answer_chunk = chunk['answer']
            answer += answer_chunk
            placeholder.markdown(answer)
        except Exception as e:
            pass
    return answer
        
VECTOR_STORE_PATH = 'viettel_vector_store'
DATASET_PATH = './rag_dataset/viettel_dataset.txt'
def main():
    st.title("Viettel Chatbot")
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = load_rag_pipeline(VECTOR_STORE_PATH)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").markdown(chat["message"])
        elif chat["role"] == "assistant":
            st.chat_message("assistant").markdown(chat["message"])

    user_input = st.chat_input("Nhập tin nhắn của bạn:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.chat_message("user").markdown(user_input)
        
        with st.chat_message("assistant"):
            answer = streamlit_generate_response(st.session_state.rag_chain, user_input)
        
        st.session_state.chat_history.append({"role": "assistant", "message": answer})

if __name__ == '__main__':
    setup_environment()
    main()