# HỆ THỐNG ĐÁNH GIÁ ĐỘ CHÍNH XÁC TÌM KIẾM SEMANTIC VỚI CHROMADB

## 📋 TỔNG QUAN

**Mục đích**: Đánh giá độ chính xác của hệ thống tìm kiếm semantic sử dụng ChromaDB để tìm kiếm hồ sơ ứng viên dựa trên các câu truy vấn.

**Ứng dụng**: Tìm kiếm ứng viên phù hợp, đánh giá chất lượng hệ thống, phân tích và cải thiện.

---

## 🏗️ CÔNG NGHỆ

- **ChromaDB**: Vector database lưu trữ embeddings
- **Sentence Transformers**: Model `all-MiniLM-L6-v2` tạo embeddings
- **Python 3.7+**: Ngôn ngữ lập trình

---

## 📁 CẤU TRÚC FILE

```
├── final_data.py              # Script chính - đánh giá độ chính xác
├── populate_chromadb.py      # Script nạp dữ liệu vào ChromaDB
├── random_queries.csv         # File queries cần đánh giá
├── resume_CLEANED.csv         # File dữ liệu hồ sơ gốc
├── chromadb_store/            # ChromaDB database
└── *.json                     # Các file kết quả (tự động tạo)
```

---

## 🔧 CHỨC NĂNG CHÍNH

### 1. Tìm kiếm Semantic
- Chuyển query thành embedding vector
- Tìm top 5 hồ sơ phù hợp nhất trong ChromaDB
- Trả về: Person ID, Title, Skills, Abilities, Program, Distance

### 2. Đánh giá Độ Chính Xác
- **Precision@K**: Tỷ lệ kết quả relevant trong top K
- **AP@K**: Trung bình precision tại các vị trí có kết quả relevant
- **MAP@K**: Trung bình AP@K qua tất cả queries

### 3. Lưu Trữ & Tiếp Tục
- Tự động lưu tiến trình sau mỗi query
- Hỗ trợ dừng và tiếp tục công việc

### 4. Thống Kê & Báo Cáo
- Thống kê tổng hợp (Precision@5, MAP@5)
- Phân tích theo category và difficulty
- Top queries tốt nhất/xấu nhất

---

## 🔑 CÁC HÀM CHÍNH

### 1. `search_top5(query: str)`
Tìm kiếm top 5 hồ sơ phù hợp nhất với query.

**Input**: Query text  
**Output**: List 5 kết quả với thông tin đầy đủ

---

### 2. `calculate_metrics(results, query, k, method, threshold)`
Tính toán các metrics đánh giá (Precision@K, AP@K).

**Input**: 
- `results`: Danh sách kết quả tìm kiếm
- `query`: Query text
- `k`: Số kết quả (mặc định 5)
- `method`: "distance" hoặc "relevance"
- `threshold`: Ngưỡng đánh giá

**Output**: Dictionary chứa `precision_at_k`, `ap_at_k`, `relevance_labels`, `num_relevant`

---

### 3. `precision_at_k(relevance_labels, k)`
Tính Precision@K = (Số kết quả relevant trong top K) / K

---

### 4. `average_precision_at_k(relevance_labels, k)`
Tính AP@K = Trung bình precision tại các vị trí có kết quả relevant

---

### 5. `calculate_relevance_score(query, result)`
Tính điểm relevance (0-1) kết hợp:
- Distance: 40%
- Title keywords: 20%
- Skills keywords: 25%
- Abilities keywords: 15%

---

### 6. `get_relevance_labels(results, query, method, threshold)`
Xác định relevance label (0 hoặc 1) cho mỗi kết quả:
- `method="distance"`: distance < threshold → relevant (1)
- `method="relevance"`: score >= threshold → relevant (1)

---

### 7. `process_queries()`
**Hàm chính** xử lý tất cả queries:
1. Đọc queries từ CSV
2. Tải progress (nếu có)
3. Xử lý từng query: tìm kiếm → tính metrics → lưu kết quả
4. Tính thống kê tổng hợp khi hoàn thành

---

### 8. `display_results(results, query_info, query_text)`
Hiển thị kết quả tìm kiếm với đánh giá relevant/non-relevant.

---

### 9-13. Các hàm hỗ trợ
- `load_queries()`: Đọc queries từ CSV
- `load_progress()`: Tải progress đã lưu
- `save_progress()`: Lưu progress
- `save_results()`: Lưu kết quả cuối cùng
- `extract_keywords()`: Trích xuất keywords từ text

---

## ⚙️ CẤU HÌNH

```python
BATCH_SIZE = 20                    # Số queries xử lý mỗi lần
COLLECTION_NAME = "qa_collection"  # Tên collection ChromaDB
AUTO_EVALUATION = True             # Tự động đánh giá
EVALUATION_METHOD = "distance"     # "distance" hoặc "relevance"
DISTANCE_THRESHOLD = 0.8           # Ngưỡng distance
RELEVANCE_THRESHOLD = 0.5          # Ngưỡng relevance score
```

**Model**: `all-MiniLM-L6-v2` (384 dimensions)

---

## 📊 METRICS ĐÁNH GIÁ

### Precision@K
Tỷ lệ kết quả relevant trong top K  
**Ví dụ**: Precision@5 = 0.8 → 4/5 kết quả relevant

### AP@K (Average Precision@K)
Trung bình precision tại các vị trí có kết quả relevant  
**Ý nghĩa**: Đánh giá chất lượng thứ tự sắp xếp

### MAP@K (Mean Average Precision@K)
Trung bình AP@K qua tất cả queries  
**Ý nghĩa**: Metric tổng hợp quan trọng nhất

---

## 🚀 CÁCH SỬ DỤNG

### Bước 1: Chuẩn bị dữ liệu
```bash
python populate_chromadb.py
```

### Bước 2: Chạy đánh giá
```bash
python final_data.py
```

### Bước 3: Xem kết quả
- Hiển thị trên console
- Lưu vào: `progress_final_data.json`, `search_results_data.json`, `final_results.json`

---

## 📈 KẾT QUẢ

**Đánh giá trên 140 queries:**
- Precision@5 trung bình: **90.29%**
- MAP@5: **97.14%**
- Queries đạt perfect: **120/140 (85.7%)**

**Theo Category:**
- FE: 100% | PM: 100% | NETSEC: 97.78% | BE: 93.75%

**Theo Difficulty:**
- Standard: 91.75% | Hard: 83.85%

---

## 💡 ĐIỂM MẠNH

✅ Tìm kiếm semantic hiệu quả  
✅ Đánh giá toàn diện với metrics chuẩn  
✅ Lưu trữ và tiếp tục công việc  
✅ Thống kê chi tiết theo nhiều tiêu chí  
✅ Tự động hóa đánh giá  
✅ Xử lý theo batch

---

## ⚠️ HẠN CHẾ & HƯỚNG PHÁT TRIỂN

**Hạn chế:**
- Model embedding nhỏ
- Chỉ tìm kiếm top 5
- Thiếu re-ranking
- Metrics hạn chế (thiếu Recall@K, NDCG@K)

**Hướng phát triển:**
- Cải thiện model (model lớn hơn, fine-tune)
- Thêm re-ranking với cross-encoder
- Mở rộng metrics (Recall@K, NDCG@K)
- Tăng số lượng kết quả (top 10, top 20)

---

## 📝 KẾT LUẬN

Hệ thống đánh giá độ chính xác tìm kiếm semantic hoàn chỉnh, đạt độ chính xác cao (Precision@5 = 90.29%, MAP@5 = 97.14%). Có thể áp dụng trong tuyển dụng, e-commerce, Q&A, tìm kiếm tài liệu.
