# 🤖 Chatbot Tra cứu Pháp Luật Hình Sự

Đây là một ứng dụng chatbot sử dụng công nghệ RAG (Retrieval-Augmented Generation) để trả lời các câu hỏi liên quan đến pháp luật hình sự Việt Nam. Ứng dụng được xây dựng bằng Streamlit và sử dụng các công nghệ AI tiên tiến để cung cấp thông tin chính xác từ các văn bản pháp luật.

## 🚀 Tính năng

- Tra cứu thông tin pháp luật hình sự thông qua giao diện chat thân thiện
- Trả lời dựa trên nguồn dữ liệu đáng tin cậy từ văn bản pháp luật
- Hỗ trợ tiếng Việt
- Trích dẫn điều luật trong câu trả lời
- Giao diện web dễ sử dụng

## 🛠️ Công nghệ sử dụng

- **Frontend**: Streamlit
- **Vector Database**: Chroma DB
- **Embedding Model**: dangvantuan/vietnamese-document-embedding
- **LLM**: Together AI (Llama-3.3-70B-Instruct-Turbo-Free)
- **Framework**: LangChain

## 📋 Yêu cầu hệ thống

- Python 3.8+
- CUDA-compatible GPU (khuyến nghị) hoặc CPU
- Together AI API key

## 🔧 Cài đặt

1. Clone repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# hoặc
.\env\Scripts\activate  # Windows
```

3. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

4. Tạo file `.env` và thêm Together AI API key:
```
TOGETHER_API_KEY=your_api_key_here
```

## 🚀 Chạy ứng dụng

1. Đảm bảo đã tạo vectorstore bằng cách chạy script `db.ipynb`

2. Khởi chạy ứng dụng:
```bash
streamlit run app.py
```

3. Truy cập ứng dụng tại địa chỉ: http://localhost:8501

## 📚 Cấu trúc dự án

```
.
├── app.py              # Ứng dụng Streamlit chính
├── main.ipynb         # Notebook chính
├── db.ipynb           # Script tạo vectorstore
├── luat_hinh_su.pdf   # Tài liệu pháp luật nguồn
├── vectorstore/       # Thư mục chứa vector database
└── env/              # Môi trường ảo Python
```

## 🤝 Demo

![image](https://github.com/user-attachments/assets/939410fb-7716-4d1e-a18e-6b9eac54f6c3)


## 📝 License

[MIT License](LICENSE) 
