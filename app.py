import streamlit as st
import os
from dotenv import load_dotenv
# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Together
from langchain_core.runnables import RunnableParallel
import torch # Import thư viện torch để kiểm tra CUDA

# --- Cấu hình Streamlit Page ---
st.set_page_config(page_title="Luật hình sự RAG Chatbot", page_icon="⚖️")
st.title("🤖 Chatbot Tra cứu Pháp Luật Hình Sự")
st.info("Tôi là một trợ lý AI, có thể cung cấp thông tin dựa trên các văn bản pháp luật hình sự. Vui lòng nhập câu hỏi của bạn bên dưới.")

# --- Load Environment Variables ---
# Load .env file at the start
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Check for API Key
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY không được tìm thấy trong biến môi trường. Vui lòng tạo file .env hoặc thiết lập biến môi trường.")
    st.stop() # Dừng ứng dụng nếu không có API key

# --- Khởi tạo các thành phần RAG với Caching (Nested Calls) ---

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model, trying CUDA first then CPU."""
    model_name = "dangvantuan/vietnamese-document-embedding"
    # Determine device
    # Sử dụng torch.cuda.is_available() để kiểm tra chính xác
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Đang tải embedding model '{model_name}' trên thiết bị: {device}...")

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'trust_remote_code': True,
                          'device': device}
        )
        st.success(f"Embedding model '{model_name}' đã tải thành công trên {device}.")
        return embedding_model
    except Exception as e:
        st.error(f"Lỗi khi tải embedding model '{model_name}' trên {device}: {e}")
        # Không dừng ở đây nếu fallback CPU có thể hoạt động,
        # nhưng trong hàm này, chúng ta đã chọn thiết bị, nên lỗi là lỗi nghiêm trọng hơn
        st.stop() # Dừng nếu tải embedding thất bại

@st.cache_resource
def get_vectorstore_retriever():
    """
    Loads the Chroma vector store and creates the retriever.
    Calls get_embedding_model() internally.
    """
    PERSIST_DIRECTORY = 'vectorstore/db_chroma'
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Thư mục vectorstore không tồn tại tại {PERSIST_DIRECTORY}.")
        st.error("Vui lòng chạy script tạo vectorstore (ví dụ: db_setup.py) trước.")
        st.stop() # Dừng ứng dụng nếu không tìm thấy vectorstore

    st.write(f"Đang tải vectorstore từ {PERSIST_DIRECTORY}...")
    # *** Gọi hàm được cache get_embedding_model() bên trong ***
    embedding_model = get_embedding_model()

    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        st.success(f"Vectorstore đã tải thành công từ {PERSIST_DIRECTORY}")
        # Tạo retriever
        retriever = vectorstore.as_retriever(
            search_type='mmr', # hoặc 'similarity'
            search_kwargs={'k': 3, 'lambda_mult': 0.7}
        )
        st.success("Retriever đã khởi tạo.")
        return retriever
    except Exception as e:
        st.error(f"Lỗi khi load vectorstore từ {PERSIST_DIRECTORY}: {e}")
        st.error("Vui lòng kiểm tra lại thư mục và nội dung của vectorstore.")
        st.stop() # Dừng nếu load vectorstore thất bại

@st.cache_resource
def get_llm(api_key):
    """Initializes the Language Model."""
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" # Đặt tên model vào biến
    st.write(f"Đang khởi tạo LLM model: {model_name}...")
    try:
        llm = Together(
            model=model_name,
            temperature=0.5,
            max_tokens=1024, # Tăng max_tokens lên một chút cho câu trả lời dài hơn
            together_api_key=api_key
        )
        st.success(f"LLM đã khởi tạo với model: {llm.model}")
        return llm
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo LLM model '{model_name}': {e}")
        st.error("Vui lòng kiểm tra TOGETHER_API_KEY và kết nối mạng.")
        st.stop() # Dừng nếu LLM thất bại

@st.cache_resource
def get_rag_chain(api_key):
    """
    Sets up the RAG chain.
    Calls get_vectorstore_retriever() and get_llm() internally.
    """
    st.write("Đang thiết lập RAG chain...")
    # *** Gọi các hàm được cache khác bên trong ***
    retriever = get_vectorstore_retriever()
    llm = get_llm(api_key) # Truyền API key vào đây

    TEMPLATE = """Bạn là một trợ lý hữu ích, chuyên cung cấp thông tin dựa trên văn bản pháp luật.
Sử dụng CHỈ thông tin trong ngữ cảnh (context) được cung cấp dưới đây để trả lời câu hỏi của người dùng.
Tổng hợp và tóm tắt thông tin liên quan từ ngữ cảnh một cách súc tích và trực tiếp, có trích dẫn điều luật.
Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy nói "Tôi không tìm thấy thông tin liên quan trong tài liệu được cung cấp."
KHÔNG sử dụng bất kỳ thông tin nào bên ngoài ngữ cảnh được cung cấp.
Bắt đầu câu trả lời ngay sau nhãn "Answer:".

Context: {context}
Question: {question}
Answer:"""
    promt_template = PromptTemplate.from_template(TEMPLATE)
    st.success("Prompt template đã tạo.")

    # Xây dựng Chuỗi RAG (LangChain Expression Language)
    chain = (
        RunnableParallel( # Sử dụng RunnableParallel để chuẩn bị input cho prompt
            context=retriever,
            question=RunnablePassthrough()
        )
        | promt_template
        | llm
        | StrOutputParser()
    )
    st.success("RAG chain đã xây dựng.")
    return chain

# --- Load the RAG Chain (will trigger loading all dependencies via caching) ---
# Chỉ cần gọi hàm get_rag_chain() ở đây.
# Streamlit sẽ tự động gọi các hàm phụ thuộc (get_vectorstore_retriever, get_llm, get_embedding_model)
# nhờ vào decorator @st.cache_resource.
with st.spinner("Đang tải các thành phần RAG (Embedding, Vectorstore, LLM, Chain)..."):
     # Truyền các tham số cần thiết cho các hàm được gọi bên trong
     rag_chain = get_rag_chain(TOGETHER_API_KEY)

# --- Giao diện người dùng ---
st.subheader("Đặt câu hỏi của bạn:")

user_question = st.text_area("Nhập câu hỏi về pháp luật:", key="question_input", height=150) # Bạn có thể chỉnh số 150 tùy ý

if st.button("Tìm kiếm"):
    if user_question:
        with st.spinner("Đang xử lý..."):
            try:
                response = rag_chain.invoke(user_question)
                # Loại bỏ tiền tố "Answer:" nếu có, xử lý cả in hoa/thường
                if isinstance(response, str) and response.strip().lower().startswith("answer:"):
                     response = response.strip()[len("answer:"):].strip()
                st.subheader("Câu trả lời:")
                st.write(response)
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                st.exception(e) # Hiển thị traceback chi tiết trong console Streamlit (hữu ích khi debug)

    else:
        st.warning("Vui lòng nhập câu hỏi.")

# Tùy chọn: Thêm thông tin về nguồn
st.markdown("---")
st.markdown("""
<small>Ứng dụng này sử dụng Embedding Model [dangvantuan/vietnamese-document-embedding](https://huggingface.co/dangvantuan/vietnamese-document-embedding), Vectorstore [Chroma DB](https://www.trychroma.com/) và LLM từ [Together AI](https://together.ai/).
Dữ liệu được lấy từ các văn bản pháp luật đã được tiền xử lý.</small>
""", unsafe_allow_html=True)