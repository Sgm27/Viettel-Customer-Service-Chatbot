import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Setup Environment
def setup_environment():
    from dotenv import load_dotenv
    import os
    import warnings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")
    load_dotenv()

# Load Dataset and Chunking
def load_and_chunking(DATASET_PATH, max_chunk_size=2000, chunk_overlap=500):
    markdown_content = ''
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True  
    )
    chunks = text_splitter.split_documents(md_header_splits)
    return chunks

# Embedding and Vector Store Setup
def setup_vector_store(chunks):
    embeddings = OllamaEmbeddings(model='bge-m3', base_url="http://localhost:11434")
    sample_vetor = embeddings.embed_query("sample")
    sample_index = faiss.IndexFlatL2(len(sample_vetor))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=sample_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

# Get Retriever
def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs={'k': 5}
    )

    return retriever

# Create RAG Chain
def create_rag_chain(retriever, model_name = 'llama3.2'):
    contextualize_q_system_prompt = '''
        Bạn được cung cấp các câu hỏi trước đó và ngữ cảnh của chúng. Hãy sử dụng thông tin này để trả lời câu hỏi tiếp theo.
    '''
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    model = ChatOllama(
        model = model_name,
        temperature = 0.3,
        base_url= "http://localhost:11434"
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """Bạn tên là Hải Vân, là một chatbot chăm sóc khách hàng chuyên nghiệp của Viettel. Nhiệm vụ của bạn là hỗ trợ khách hàng một cách nhanh chóng, chính xác và thân thiện. Bạn cần giải đáp các câu hỏi về dịch vụ di động, internet, truyền hình, gói cước, khuyến mãi, thanh toán và các vấn đề kỹ thuật khác.
            Hãy sử dụng giọng điệu chuyên nghiệp, lịch sự nhưng gần gũi để tạo cảm giác thoải mái cho khách hàng. Khi cần thiết, hãy đề xuất giải pháp phù hợp hoặc hướng dẫn khách hàng liên hệ tổng đài hoặc trung tâm hỗ trợ. Nếu không có đủ thông tin để trả lời, hãy xin lỗi khách hàng và đề xuất cách họ có thể nhận được hỗ trợ từ Viettel.
            Luôn ghi nhớ mục tiêu: Giúp khách hàng có trải nghiệm tốt nhất với dịch vụ của Viettel!"

            ### Ngữ cảnh : {context}

            ### Trả lời :
            """


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history", return_messages=False),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

# Load Vector Store
def load_vector_store(VECTOR_STORE_PATH):
    embedding = OllamaEmbeddings(model='bge-m3', base_url="http://localhost:11434")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)
    return vector_store

# Generate Response
def generate_response(rag_chain, input_text, session_id = "default"):
    for chunk in rag_chain.stream(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    ):
        try:
            print(chunk['answer'], end='', flush=True)
        except:
            pass

# Create RAG Pipeline from Scratch
def create_rag_pipeline_from_scratch(DATASET_PATH):
    chunks = load_and_chunking(DATASET_PATH=DATASET_PATH)
    vector_store = setup_vector_store(chunks)
    retriever = get_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)
    return rag_chain

# Load RAG Pipeline
def load_rag_pipeline(VECTOR_STORE_PATH):
    vector_store = load_vector_store(VECTOR_STORE_PATH=VECTOR_STORE_PATH)
    retriever = get_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)
    return rag_chain