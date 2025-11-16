# -*- coding: utf-8 -*-
"""
Script đánh giá độ chính xác của hệ thống tìm kiếm
- Đọc queries từ random_queries.csv
- Chạy search_top5 cho mỗi query
- Tính Precision@K, AP@K, MAP@K dựa trên distance hoặc relevance score
- Lưu progress để có thể tiếp tục sau
"""
import csv  # Đọc/ghi file CSV chứa queries
import json  # Đọc/ghi file JSON chứa progress và kết quả
import re  # Xử lý regex để trích xuất keywords
from pathlib import Path  # Xử lý đường dẫn file cross-platform
from typing import List, Dict, Any, Optional  # Type hints cho code rõ ràng hơn
from sentence_transformers import SentenceTransformer  # Model embedding để chuyển text thành vector
import chromadb  # Database vector để lưu trữ và tìm kiếm embeddings

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ============================================================================

# Thư mục gốc của project (thư mục chứa file này)
BASE_DIR = Path(__file__).resolve().parent

# File chứa danh sách queries cần đánh giá (CSV format)
QUERIES_FILE = BASE_DIR / "random_queries.csv"

# File lưu tiến độ xử lý để có thể tiếp tục sau khi dừng
PROGRESS_FILE = BASE_DIR / "progress_final_data.json"

# File lưu kết quả đánh giá cuối cùng (metrics cho mỗi query)
RESULTS_FILE = BASE_DIR / "final_results.json"

# File lưu kết quả tìm kiếm chi tiết để đánh giá sau
SEARCH_RESULTS_FILE = BASE_DIR / "search_results_data.json"

# Tên collection trong ChromaDB chứa dữ liệu hồ sơ
COLLECTION_NAME = "qa_collection"

# Số lượng queries xử lý mỗi lần chạy script (để tránh xử lý quá nhiều một lúc)
BATCH_SIZE = 20

# Chế độ đánh giá: True = tự động đánh giá dựa trên threshold, False = yêu cầu người dùng nhập
AUTO_EVALUATION = True

# Phương pháp đánh giá: "distance" (chỉ dùng distance) hoặc "relevance" (dùng relevance score kết hợp)
EVALUATION_METHOD = "distance"

# Ngưỡng distance: nếu distance < threshold thì kết quả được coi là relevant
# Distance càng nhỏ = càng giống = càng phù hợp
DISTANCE_THRESHOLD = 0.8

# Ngưỡng relevance score: nếu score >= threshold thì kết quả được coi là relevant
# Relevance score càng cao = càng phù hợp (0-1)
RELEVANCE_THRESHOLD = 0.5

# ============================================================================
# KHỞI TẠO CHROMADB VÀ MODEL EMBEDDING
# ============================================================================

# Khởi tạo ChromaDB client với persistent storage (dữ liệu lưu trên disk)
# Dữ liệu được lưu trong thư mục chromadb_store
client = chromadb.PersistentClient(path=str(BASE_DIR / "chromadb_store"))

# Lấy hoặc tạo collection chứa dữ liệu hồ sơ
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Khởi tạo model embedding để chuyển đổi text thành vector
# all-MiniLM-L6-v2 là model nhẹ, nhanh, phù hợp cho semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')


def search_top5(query: str) -> List[Dict[str, Any]]:
    """
    Tìm kiếm top 5 hồ sơ phù hợp nhất với query sử dụng semantic search.
    
    Quy trình:
    1. Chuyển đổi query text thành embedding vector
    2. Tìm kiếm trong ChromaDB collection để tìm các vector gần nhất
    3. Trả về top 5 kết quả kèm metadata và distance
    
    Args:
        query: Câu truy vấn cần tìm kiếm (ví dụ: "Tìm developer Python có kinh nghiệm Django")
    
    Returns:
        List chứa tối đa 5 dict, mỗi dict có:
            - person_id: ID của hồ sơ
            - title: Chức danh/vai trò
            - skills: Danh sách kỹ năng
            - abilities: Khả năng/năng lực
            - program: Chương trình đào tạo/bằng cấp
            - distance: Khoảng cách semantic (càng nhỏ càng giống)
    """
    # Chuyển đổi query text thành embedding vector
    # convert_to_tensor=False: trả về numpy array thay vì tensor
    # [0].tolist(): lấy vector đầu tiên và chuyển thành list Python
    q_emb = model.encode([query], convert_to_tensor=False)[0].tolist()
    
    # Tìm kiếm trong ChromaDB collection
    # query_embeddings: danh sách embedding vectors cần tìm
    # n_results: số lượng kết quả tối đa (5)
    # include: chỉ định các thông tin cần trả về (metadata và distances)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "distances"],
    )
    
    # Khởi tạo danh sách kết quả
    items: List[Dict[str, Any]] = []
    
    # Lấy metadata của các kết quả (thông tin chi tiết về hồ sơ)
    # results.get("metadatas") trả về list of lists, lấy phần tử đầu tiên
    # Nếu không có thì dùng [[]] và lấy [0] để tránh lỗi
    metas_list = (results.get("metadatas") or [[]])[0]
    
    # Lấy distances (khoảng cách semantic) của các kết quả
    distance_list = (results.get("distances") or [[]])[0]
    
    # Lấy IDs của các kết quả (ids luôn được trả về, không cần include)
    ids_list = (results.get("ids") or [[]])[0]
    
    # Duyệt qua từng kết quả và tổng hợp thông tin
    for idx, meta in enumerate(metas_list):
        # Lấy distance tương ứng với kết quả này (nếu có)
        distance = distance_list[idx] if idx < len(distance_list) else None
        
        # Lấy person_id tương ứng (nếu có)
        person_id = ids_list[idx] if idx < len(ids_list) else None
        
        # Tạo dict chứa thông tin hồ sơ
        items.append({
            "person_id": person_id,
            "title": meta.get("title", ""),  # Chức danh/vai trò
            "skills": meta.get("skills", ""),  # Kỹ năng
            "abilities": meta.get("abilities", ""),  # Khả năng
            "program": meta.get("program", ""),  # Chương trình đào tạo
            "distance": distance,  # Khoảng cách semantic
        })
    
    return items


def load_queries() -> List[Dict[str, str]]:
    """
    Đọc tất cả queries từ file CSV.
    
    File CSV phải có các cột:
    - query_id: ID duy nhất của query
    - query_text: Nội dung câu truy vấn
    - category: Phân loại query (ví dụ: skills, experience, education)
    - target_person_id: ID của hồ sơ đúng (nếu có)
    - difficulty: Độ khó của query (ví dụ: easy, medium, hard)
    
    Returns:
        List các dict, mỗi dict chứa thông tin một query
    
    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại
    """
    queries = []
    
    # Kiểm tra file có tồn tại không
    if not QUERIES_FILE.exists():
        raise FileNotFoundError(f"File not found: {QUERIES_FILE}")
    
    # Mở file CSV và đọc từng dòng
    # encoding="utf-8": hỗ trợ tiếng Việt và các ký tự đặc biệt
    # newline="": để csv.DictReader xử lý newline đúng cách
    with QUERIES_FILE.open("r", encoding="utf-8", newline="") as f:
        # DictReader tự động đọc header và map các cột thành keys
        reader = csv.DictReader(f)
        
        # Đọc từng dòng và tạo dict cho mỗi query
        for row in reader:
            queries.append({
                "query_id": row.get("query_id", ""),  # ID của query
                "query_text": row.get("query_text", ""),  # Nội dung truy vấn
                "category": row.get("category", ""),  # Phân loại
                "target_person_id": row.get("target_person_id", ""),  # ID hồ sơ đúng (nếu có)
                "difficulty": row.get("difficulty", ""),  # Độ khó
            })
    
    return queries


def load_progress() -> Dict[str, Any]:
    """
    Tải tiến độ xử lý đã lưu từ file (nếu có).
    
    Progress file chứa:
    - last_processed_index: Chỉ số query cuối cùng đã xử lý
    - results: Danh sách kết quả đã xử lý
    
    Returns:
        Dict chứa progress, hoặc dict mặc định nếu chưa có file
    """
    # Kiểm tra file progress có tồn tại không
    if PROGRESS_FILE.exists():
        # Đọc file JSON và trả về dữ liệu
        with PROGRESS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    # Nếu chưa có file, trả về progress mặc định (bắt đầu từ đầu)
    return {
        "last_processed_index": 0,  # Chưa xử lý query nào
        "results": []  # Chưa có kết quả nào
    }


def save_progress(progress: Dict[str, Any]):
    """
    Lưu tiến độ xử lý vào file để có thể tiếp tục sau.
    
    Args:
        progress: Dict chứa:
            - last_processed_index: Chỉ số query cuối cùng đã xử lý
            - results: Danh sách kết quả đã xử lý
    """
    # Ghi dữ liệu vào file JSON
    # ensure_ascii=False: cho phép lưu ký tự tiếng Việt
    # indent=2: format đẹp, dễ đọc
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_results(results: List[Dict[str, Any]]):
    """
    Lưu kết quả đánh giá cuối cùng vào file.
    
    Args:
        results: List các dict, mỗi dict chứa:
            - query_id, query_text, category, difficulty
            - precision_at_5, ap_at_5, relevance_labels, num_relevant
            - search_results: kết quả tìm kiếm chi tiết
    """
    # Ghi kết quả vào file JSON
    # ensure_ascii=False: hỗ trợ tiếng Việt
    # indent=2: format đẹp
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_search_results() -> List[Dict[str, Any]]:
    """
    Tải kết quả tìm kiếm đã lưu từ file (nếu có).
    
    File này chứa kết quả tìm kiếm chi tiết cho mỗi query,
    dùng để đánh giá sau hoặc phân tích.
    
    Returns:
        List các dict chứa kết quả tìm kiếm, hoặc list rỗng nếu chưa có file
    """
    # Kiểm tra file có tồn tại không
    if SEARCH_RESULTS_FILE.exists():
        # Đọc file JSON
        with SEARCH_RESULTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    # Nếu chưa có file, trả về list rỗng
    return []


def save_search_results(search_results_data: List[Dict[str, Any]]):
    """
    Lưu kết quả tìm kiếm vào file để đánh giá sau.
    
    Args:
        search_results_data: List các dict, mỗi dict chứa:
            - query_id, query_text, category, target_person_id, difficulty
            - search_results: top 5 kết quả tìm kiếm
            - timestamp: thời gian tìm kiếm (nếu có)
    """
    # Ghi kết quả tìm kiếm vào file JSON
    # ensure_ascii=False: hỗ trợ tiếng Việt
    # indent=2: format đẹp
    with SEARCH_RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(search_results_data, f, ensure_ascii=False, indent=2)


def display_results(results: List[Dict[str, Any]], query_info: Dict[str, str], query_text: str = ""):
    """
    Hiển thị kết quả tìm kiếm cho người dùng với đánh giá đúng/khớp.
    
    Hàm này hiển thị:
    - Thông tin query (ID, nội dung, category, difficulty)
    - Top 5 kết quả tìm kiếm với đầy đủ thông tin
    - Đánh dấu kết quả nào là relevant/không relevant dựa trên threshold
    - Distance và relevance score (nếu có)
    
    Args:
        results: List các kết quả tìm kiếm (tối đa 5)
        query_info: Dict chứa thông tin query (query_id, query_text, category, difficulty)
        query_text: Nội dung query (dùng để tính relevance score nếu cần)
    """
    # In header với thông tin query
    print("\n" + "="*80)
    print(f"QUERY THÔNG TIN")
    print("="*80)
    print(f"Query ID: {query_info['query_id']}")
    print(f"Query: {query_info['query_text']}")
    print(f"Category: {query_info['category']} | Difficulty: {query_info['difficulty']}")
    print("\n" + "-"*80)
    print("TOP 5 KẾT QUẢ TÌM KIẾM")
    print("-"*80)
    
    # Xác định threshold và mô tả phương pháp đánh giá
    # Dựa trên EVALUATION_METHOD để chọn threshold phù hợp
    if EVALUATION_METHOD == "distance":
        threshold = DISTANCE_THRESHOLD  # Ngưỡng distance
        method_desc = f"distance < {threshold}"  # Mô tả: distance nhỏ hơn threshold = relevant
    else:
        threshold = RELEVANCE_THRESHOLD  # Ngưỡng relevance score
        method_desc = f"relevance >= {threshold}"  # Mô tả: relevance lớn hơn hoặc bằng threshold = relevant
    
    # Hiển thị tiêu chí đánh giá cho người dùng
    print(f"\n📊 Tiêu chí đánh giá: {method_desc}")
    if EVALUATION_METHOD == "distance":
        print("   💡 Distance càng NHỎ → càng giống → càng ĐÚNG")
    else:
        print("   💡 Relevance score càng CAO → càng phù hợp → càng ĐÚNG")
    print("\n" + "-"*80)
    
    # Duyệt qua từng kết quả và hiển thị chi tiết
    for idx, result in enumerate(results, 1):
        # Lấy thông tin từ kết quả
        person_id = result.get('person_id', 'N/A')  # ID của hồ sơ
        distance = result.get('distance', None)  # Khoảng cách semantic
        title = result.get('title', 'N/A')  # Chức danh/vai trò
        skills = result.get('skills', 'N/A')  # Danh sách kỹ năng
        abilities = result.get('abilities', 'N/A')  # Khả năng/năng lực
        program = result.get('program', 'N/A')  # Chương trình đào tạo
        
        # Xác định xem kết quả có đúng/khớp (relevant) không
        # Dựa trên phương pháp đánh giá đã chọn
        is_relevant = False
        if EVALUATION_METHOD == "distance":
            # Nếu dùng distance: distance < threshold → relevant
            if distance is not None:
                is_relevant = distance < threshold
        else:
            # Nếu dùng relevance score: score >= threshold → relevant
            if query_text:
                score = calculate_relevance_score(query_text, result)
                is_relevant = score >= threshold
        
        # Tạo icon và text để đánh dấu kết quả
        status_icon = "✓ ĐÚNG/KHỚP" if is_relevant else "✗ KHÔNG KHỚP"
        status_color = "✓" if is_relevant else "✗"
        
        # In header cho từng kết quả
        print(f"\n{'='*80}")
        print(f"KẾT QUẢ [{idx}/5] - {status_icon}")
        print(f"{'='*80}")
        print(f"Person ID: {person_id}")
        
        # Hiển thị distance và relevance score (nếu có)
        if distance is not None:
            relevance_info = f"Distance: {distance:.4f}"
            if EVALUATION_METHOD == "distance":
                # Chỉ hiển thị distance với dấu so sánh với threshold
                relevance_info += f" {'✓' if is_relevant else '✗'} {'< ' if is_relevant else '≥ '}{threshold}"
            else:
                # Hiển thị cả distance và relevance score
                if query_text:
                    score = calculate_relevance_score(query_text, result)
                    relevance_info += f" | Relevance: {score:.3f} {'✓' if is_relevant else '✗'} {'≥ ' if is_relevant else '< '}{threshold}"
            print(f"{relevance_info}")
        
        # Hiển thị thông tin chi tiết của hồ sơ
        print(f"\n📋 Title/Role:")
        print(f"   {title}")
        
        print(f"\n🛠️  Skills:")
        # Hiển thị skills, tự động chia thành nhiều dòng nếu quá dài
        # Để tránh text quá dài làm khó đọc
        if skills and skills != 'N/A':
            # Chia skills thành các từ (phân cách bởi ', ')
            words = skills.split(', ')
            line = ""  # Dòng hiện tại đang xây dựng
            for word in words:
                # Nếu thêm từ này vào sẽ vượt quá 75 ký tự, in dòng hiện tại và bắt đầu dòng mới
                if len(line) + len(word) + 2 > 75:  # +2 cho ', '
                    if line:
                        print(f"   {line.strip()}")
                    line = word + ", "  # Bắt đầu dòng mới
                else:
                    line += word + ", "  # Thêm vào dòng hiện tại
            # In dòng cuối cùng nếu còn
            if line:
                print(f"   {line.rstrip(', ')}")
        else:
            print(f"   {skills}")
        
        # Tương tự cho abilities
        print(f"\n💼 Abilities:")
        if abilities and abilities != 'N/A':
            # Chia thành các dòng 80 ký tự để dễ đọc
            words = abilities.split(', ')
            line = ""
            for word in words:
                if len(line) + len(word) + 2 > 75:
                    if line:
                        print(f"   {line.strip()}")
                    line = word + ", "
                else:
                    line += word + ", "
            if line:
                print(f"   {line.rstrip(', ')}")
        else:
            print(f"   {abilities}")
        
        # Hiển thị thông tin giáo dục
        print(f"\n🎓 Education/Program:")
        print(f"   {program}")
    
    # Kết thúc phần hiển thị
    print("\n" + "="*80)


def extract_keywords(text: str) -> set:
    """
    Trích xuất keywords từ text bằng cách loại bỏ stop words.
    
    Hàm này:
    1. Chuyển text thành lowercase
    2. Tách thành các từ (words)
    3. Loại bỏ các từ quá ngắn (< 3 ký tự)
    4. Loại bỏ các stop words (từ thường gặp, không có ý nghĩa)
    
    Args:
        text: Chuỗi text cần trích xuất keywords
    
    Returns:
        Set các keywords (từ có ý nghĩa)
    
    Ví dụ:
        "I am a Python developer" → {'python', 'developer'}
    """
    # Chuyển thành lowercase và tách thành từ bằng regex
    # r'\b\w+\b': match các từ (word boundaries)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Danh sách stop words (các từ thường gặp, không có ý nghĩa trong tìm kiếm)
    # Bao gồm: mạo từ, giới từ, động từ to be, động từ phụ, v.v.
    stop_words = {'the', 'for', 'and', 'with', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'of', 'from', 'by', 'as', 'or', 'but', 'not', 'this', 'that', 'these', 'those'}
    
    # Lọc: chỉ giữ các từ có >= 3 ký tự và không phải stop word
    keywords = {w for w in words if len(w) >= 3 and w not in stop_words}
    
    return keywords


def calculate_relevance_score(query: str, result: Dict[str, Any]) -> float:
    """
    Tính điểm relevance (0-1) kết hợp semantic similarity và keyword matching.
    
    Điểm relevance được tính dựa trên 4 thành phần:
    1. Distance (semantic similarity) - 40%
       - Dựa trên embedding vectors, đo độ tương đồng ngữ nghĩa
       - Distance càng nhỏ → similarity càng cao
    2. Keyword matching trong title - 20%
       - So khớp keywords giữa query và title
    3. Keyword matching trong skills - 25%
       - So khớp keywords giữa query và skills (quan trọng nhất)
    4. Keyword matching trong abilities - 15%
       - So khớp keywords giữa query và abilities
    
    Args:
        query: Câu truy vấn
        result: Dict chứa thông tin kết quả (distance, title, skills, abilities)
    
    Returns:
        Điểm relevance từ 0.0 đến 1.0 (càng cao càng phù hợp)
    """
    score = 0.0
    
    # 1. Distance score (semantic similarity) - 40%
    # Distance đo khoảng cách giữa embedding vectors
    # Distance càng nhỏ = càng giống nhau về mặt ngữ nghĩa
    distance = result.get('distance')
    if distance is not None:
        # Normalize distance về khoảng 0-1
        # Distance thường trong khoảng 0-2 (cosine distance)
        # Nếu distance = 0 → score = 1 (hoàn toàn giống)
        # Nếu distance = 2 → score = 0 (hoàn toàn khác)
        distance_score = max(0, 1 - (distance / 2.0))
        score += distance_score * 0.4  # Trọng số 40%
    
    # 2-4. Keyword matching (so khớp từ khóa)
    # Trích xuất keywords từ query
    query_keywords = extract_keywords(query)
    
    # 2. Title matching - 20%
    # So khớp keywords giữa query và title (chức danh/vai trò)
    title = result.get('title', '').lower()
    title_keywords = extract_keywords(title)
    # Tính tỷ lệ keywords trùng: số keywords chung / tổng số keywords trong query
    title_match = len(query_keywords & title_keywords) / max(len(query_keywords), 1)
    score += title_match * 0.2  # Trọng số 20%
    
    # 3. Skills matching - 25% (quan trọng nhất trong keyword matching)
    # So khớp keywords giữa query và skills
    skills = result.get('skills', '').lower()
    skills_keywords = extract_keywords(skills)
    skills_match = len(query_keywords & skills_keywords) / max(len(query_keywords), 1)
    score += skills_match * 0.25  # Trọng số 25%
    
    # 4. Abilities matching - 15%
    # So khớp keywords giữa query và abilities
    abilities = result.get('abilities', '').lower()
    abilities_keywords = extract_keywords(abilities)
    abilities_match = len(query_keywords & abilities_keywords) / max(len(query_keywords), 1)
    score += abilities_match * 0.15  # Trọng số 15%
    
    # Đảm bảo score không vượt quá 1.0
    return min(1.0, score)


def get_relevance_labels(results: List[Dict[str, Any]], query: str, 
                         method: str = "distance", threshold: float = 0.8) -> List[int]:
    """
    Xác định relevance label (0 hoặc 1) cho mỗi kết quả tìm kiếm.
    
    Relevance label là nhãn nhị phân:
    - 1: Kết quả phù hợp (relevant) với query
    - 0: Kết quả không phù hợp (non-relevant) với query
    
    Có 2 phương pháp xác định:
    1. Distance-based: Dựa trên khoảng cách semantic
    2. Relevance score-based: Dựa trên điểm relevance (kết hợp distance + keywords)
    
    Args:
        results: List các kết quả tìm kiếm
        query: Nội dung query (dùng để tính relevance score nếu method="relevance")
        method: 
            - "distance": Chỉ dùng distance để đánh giá
            - "relevance": Dùng relevance score (kết hợp distance + keyword matching)
        threshold: 
            - Nếu method="distance": distance threshold (thường 0.6-1.0)
              distance < threshold → relevant
            - Nếu method="relevance": relevance score threshold (0-1)
              score >= threshold → relevant
    
    Returns:
        List các nhãn nhị phân [0, 1, 0, 1, ...] tương ứng với từng kết quả
    """
    labels = []
    
    # Duyệt qua từng kết quả và xác định relevance
    for result in results:
        is_relevant = False
        
        if method == "distance":
            # Phương pháp 1: Chỉ dùng distance
            # Distance càng nhỏ = càng giống nhau về mặt ngữ nghĩa = càng phù hợp
            distance = result.get('distance')
            if distance is not None:
                # Nếu distance nhỏ hơn threshold → kết quả là relevant
                is_relevant = distance < threshold
        elif method == "relevance":
            # Phương pháp 2: Dùng relevance score (kết hợp distance + keyword matching)
            # Tính relevance score cho kết quả này
            score = calculate_relevance_score(query, result)
            # Nếu score lớn hơn hoặc bằng threshold → kết quả là relevant
            is_relevant = score >= threshold
        
        # Chuyển đổi boolean thành nhãn nhị phân (1 hoặc 0)
        labels.append(1 if is_relevant else 0)
    
    return labels


def precision_at_k(relevance_labels: List[int], k: int) -> float:
    """
    Tính Precision@K - tỷ lệ kết quả relevant trong top K.
    
    Precision@K = (Số kết quả relevant trong top K) / K
    
    Metric này đo độ chính xác của top K kết quả đầu tiên.
    Ví dụ: Precision@5 = 0.8 nghĩa là 80% (4/5) kết quả trong top 5 là relevant.
    
    Args:
        relevance_labels: List các nhãn nhị phân [0, 1, 0, 1, ...]
                         1 = relevant, 0 = non-relevant
        k: Số kết quả đầu tiên cần xem xét (ví dụ: 5)
    
    Returns:
        Precision@K trong khoảng 0.0 đến 1.0
        - 1.0: Tất cả top K đều relevant
        - 0.0: Không có kết quả nào relevant trong top K
    
    Ví dụ:
        relevance_labels = [1, 1, 0, 1, 0], k=5
        → Precision@5 = 3/5 = 0.6
    """
    # Xử lý edge cases
    if k == 0:
        return 0.0
    
    # Lấy top K labels
    top_k_labels = relevance_labels[:k]
    if not top_k_labels:
        return 0.0
    
    # Đếm số kết quả relevant trong top K
    relevant_count = sum(top_k_labels)
    
    # Precision = số relevant / tổng số kết quả trong top K
    return relevant_count / len(top_k_labels)


def average_precision_at_k(relevance_labels: List[int], k: int) -> float:
    """
    Tính Average Precision@K (AP@K) - trung bình precision tại các vị trí relevant.
    
    AP@K đo chất lượng ranking bằng cách:
    - Tính precision tại mỗi vị trí có kết quả relevant
    - Lấy trung bình các precision đó
    
    Công thức:
    AP@K = (1/R) * sum(P@i for i where result i is relevant)
    Trong đó:
    - R = tổng số kết quả relevant trong top K
    - P@i = precision tại vị trí i (số relevant từ đầu đến i / i)
    
    AP@K tốt hơn Precision@K vì nó phạt các kết quả relevant xuất hiện ở vị trí thấp.
    
    Args:
        relevance_labels: List các nhãn nhị phân [0, 1, 0, 1, ...]
        k: Số kết quả đầu tiên cần xem xét
    
    Returns:
        AP@K trong khoảng 0.0 đến 1.0
    
    Ví dụ:
        relevance_labels = [1, 0, 1, 0, 1], k=5
        - Vị trí 1: relevant → P@1 = 1/1 = 1.0
        - Vị trí 3: relevant → P@3 = 2/3 = 0.667
        - Vị trí 5: relevant → P@5 = 3/5 = 0.6
        - AP@5 = (1.0 + 0.667 + 0.6) / 3 = 0.756
    """
    # Xử lý edge cases
    if k == 0:
        return 0.0
    
    # Lấy top K labels
    top_k_labels = relevance_labels[:k]
    if not top_k_labels:
        return 0.0
    
    # Tổng số kết quả relevant trong top K
    total_relevant = sum(top_k_labels)
    if total_relevant == 0:
        # Nếu không có kết quả relevant nào → AP@K = 0
        return 0.0
    
    # Tính precision tại mỗi vị trí có kết quả relevant
    ap_sum = 0.0  # Tổng các precision
    relevant_found = 0  # Số relevant đã tìm thấy từ đầu đến vị trí hiện tại
    
    # Duyệt qua từng vị trí trong top K
    for i, label in enumerate(top_k_labels, 1):  # i bắt đầu từ 1
        if label == 1:  # Kết quả tại vị trí i là relevant
            relevant_found += 1
            # Precision tại vị trí i = số relevant từ đầu đến i / i
            precision_at_i = relevant_found / i
            ap_sum += precision_at_i
    
    # AP@K = trung bình các precision tại các vị trí relevant
    return ap_sum / total_relevant


def calculate_metrics(results: List[Dict[str, Any]], query: str,
                     k: int = 5, method: str = "distance", threshold: float = 0.8) -> Dict[str, float]:
    """
    Tính các metrics đánh giá chất lượng tìm kiếm: Precision@K và AP@K.
    
    Hàm này:
    1. Xác định relevance labels cho các kết quả
    2. Tính Precision@K (tỷ lệ relevant trong top K)
    3. Tính AP@K (trung bình precision tại các vị trí relevant)
    
    Args:
        results: List các kết quả tìm kiếm (thường là top 5)
        query: Nội dung query (dùng để tính relevance score nếu method="relevance")
        k: Số kết quả đầu tiên cần đánh giá (mặc định 5)
        method: 
            - "distance": Chỉ dùng distance để xác định relevant
            - "relevance": Dùng relevance score (kết hợp distance + keywords)
        threshold: 
            - Nếu method="distance": distance threshold (thường 0.6-1.0)
              distance < threshold → relevant
            - Nếu method="relevance": relevance score threshold (0-1)
              score >= threshold → relevant
    
    Returns:
        Dict chứa:
            - precision_at_k: Precision@K (0.0-1.0)
            - ap_at_k: Average Precision@K (0.0-1.0)
            - relevance_labels: List nhãn nhị phân [0, 1, 0, ...]
            - num_relevant: Số lượng kết quả relevant trong top K
    """
    # Bước 1: Xác định relevance labels cho từng kết quả
    # Mỗi kết quả được gán nhãn 1 (relevant) hoặc 0 (non-relevant)
    relevance_labels = get_relevance_labels(results, query, method, threshold)
    
    # Bước 2: Tính Precision@K
    # Tỷ lệ kết quả relevant trong top K
    p_at_k = precision_at_k(relevance_labels, k)
    
    # Bước 3: Tính AP@K (Average Precision@K)
    # Trung bình precision tại các vị trí có kết quả relevant
    ap_at_k = average_precision_at_k(relevance_labels, k)
    
    # Trả về tất cả metrics
    return {
        'precision_at_k': p_at_k,  # Precision@K
        'ap_at_k': ap_at_k,  # Average Precision@K
        'relevance_labels': relevance_labels,  # Nhãn của từng kết quả
        'num_relevant': sum(relevance_labels)  # Tổng số relevant
    }


def auto_evaluate_results(query: str, results: List[Dict[str, Any]], method: str = "combined", threshold: float = 0.5) -> int:
    """
    Tự động đánh giá số kết quả phù hợp.
    
    Args:
        query: Query text
        results: List of search results
        method: "distance", "keywords", hoặc "combined"
        threshold: Ngưỡng để coi là phù hợp (0-1)
    
    Returns:
        Số kết quả được đánh giá là phù hợp (0-5)
    """
    correct_count = 0
    
    for result in results:
        is_relevant = False
        
        if method == "distance":
            # Chỉ dựa trên distance
            distance = result.get('distance')
            if distance is not None:
                # Distance < threshold * 2 (vì distance thường 0-2)
                is_relevant = distance < (threshold * 2)
        
        elif method == "keywords":
            # Chỉ dựa trên keyword matching
            score = calculate_relevance_score(query, result)
            is_relevant = score >= threshold
        
        elif method == "combined":
            # Kết hợp distance và keywords
            score = calculate_relevance_score(query, result)
            is_relevant = score >= threshold
        
        if is_relevant:
            correct_count += 1
    
    return correct_count


def get_correct_count(query: str, results: List[Dict[str, Any]], auto_mode: bool = False) -> int:
    """
    Đánh giá số câu trả lời đúng.
    Nếu auto_mode=True, tự động đánh giá. Nếu False, yêu cầu người dùng nhập.
    """
    if auto_mode:
        # Tự động đánh giá
        print("\n" + "="*80)
        print("ĐÁNH GIÁ TỰ ĐỘNG")
        print("="*80)
        
        if EVALUATION_METHOD == "distance":
            print(f"Đang đánh giá tự động dựa trên DISTANCE (distance < {DISTANCE_THRESHOLD} → relevant)")
            print("  💡 Distance càng NHỎ → càng giống → càng đúng")
            print("\nDistance của từng kết quả:")
            for idx, result in enumerate(results, 1):
                distance = result.get('distance', 'N/A')
                person_id = result.get('person_id', 'N/A')
                is_relevant = distance != 'N/A' and distance < DISTANCE_THRESHOLD
                status = "✓ RELEVANT" if is_relevant else "✗ Non-relevant"
                print(f"  [{idx}] Person ID: {person_id} | Distance: {distance:.4f} {status}")
            
            # Đếm số relevant
            correct_count = sum(1 for r in results 
                              if r.get('distance') is not None and r.get('distance') < DISTANCE_THRESHOLD)
            print(f"\n✓ Tự động đánh giá: {correct_count}/5 kết quả phù hợp (distance < {DISTANCE_THRESHOLD})")
        else:
            print(f"Đang đánh giá tự động dựa trên RELEVANCE SCORE (score >= {RELEVANCE_THRESHOLD} → relevant)")
            print("  💡 Relevance score càng CAO → càng phù hợp → càng đúng")
            print("\nRelevance score của từng kết quả:")
            for idx, result in enumerate(results, 1):
                score = calculate_relevance_score(query, result)
                distance = result.get('distance', 'N/A')
                person_id = result.get('person_id', 'N/A')
                is_relevant = score >= RELEVANCE_THRESHOLD
                status = "✓ RELEVANT" if is_relevant else "✗ Non-relevant"
                print(f"  [{idx}] Person ID: {person_id} | Distance: {distance:.4f} | Relevance: {score:.3f} {status}")
            
            # Đếm số relevant
            correct_count = sum(1 for r in results 
                              if calculate_relevance_score(query, r) >= RELEVANCE_THRESHOLD)
            print(f"\n✓ Tự động đánh giá: {correct_count}/5 kết quả phù hợp (relevance >= {RELEVANCE_THRESHOLD})")
        
        # Cho phép người dùng xác nhận hoặc chỉnh sửa
        print("\nBạn có muốn chỉnh sửa kết quả này không? (Enter để chấp nhận, hoặc nhập số 0-5):")
        user_input = input(">>> ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            return -1
        elif user_input == "":
            return correct_count
        else:
            try:
                count = int(user_input)
                if 0 <= count <= 5:
                    return count
                else:
                    print("⚠️  Số không hợp lệ, sử dụng kết quả tự động.")
                    return correct_count
            except ValueError:
                print("⚠️  Không hợp lệ, sử dụng kết quả tự động.")
                return correct_count
    else:
        # Đánh giá thủ công
        print("\n" + "="*80)
        print("ĐÁNH GIÁ KẾT QUẢ")
        print("="*80)
        print("Dựa trên các tiêu chí đã gợi ý ở trên, hãy đếm số kết quả PHÙ HỢP với query.")
        print("Một kết quả được coi là PHÙ HỢP nếu:")
        print("  ✓ Có các kỹ năng/công nghệ được yêu cầu trong query")
        print("  ✓ Chức danh/vai trò phù hợp với yêu cầu")
        print("  ✓ Bằng cấp/giáo dục phù hợp (nếu query yêu cầu)")
        print("  ✓ Có độ liên quan tổng thể tốt với query")
        print("\nNhập số kết quả PHÙ HỢP (0-5):")
        while True:
            try:
                count = input(">>> ").strip()
                if count.lower() in ['exit', 'quit', 'q']:
                    return -1  # Signal để dừng
                count = int(count)
                if 0 <= count <= 5:
                    return count
                else:
                    print("⚠️  Vui lòng nhập số từ 0 đến 5!")
            except ValueError:
                print("⚠️  Vui lòng nhập một số hợp lệ!")


def process_queries():
    """
    Hàm chính xử lý tất cả queries từ file CSV và đánh giá hệ thống tìm kiếm.
    
    Quy trình:
    1. Đọc queries từ file CSV
    2. Tải progress đã lưu (nếu có) để tiếp tục từ vị trí đã dừng
    3. Xử lý từng query trong batch:
       - Tìm kiếm top 5 kết quả
       - Hiển thị kết quả cho người dùng
       - Tính metrics (Precision@5, AP@5)
       - Lưu kết quả và progress
    4. Khi hoàn thành, tính MAP@5 và các thống kê tổng hợp
    
    Hàm này hỗ trợ:
    - Xử lý theo batch (BATCH_SIZE queries mỗi lần)
    - Lưu progress để có thể tiếp tục sau khi dừng
    - Đánh giá tự động hoặc thủ công
    - Tính toán và hiển thị metrics chi tiết
    """
    # ========================================================================
    # BƯỚC 1: ĐỌC QUERIES TỪ FILE CSV
    # ========================================================================
    print("Đang đọc queries từ file...")
    all_queries = load_queries()  # Đọc tất cả queries từ CSV
    print(f"Tổng số queries: {len(all_queries)}")
    
    # ========================================================================
    # BƯỚC 2: TẢI PROGRESS ĐÃ LƯU (NẾU CÓ)
    # ========================================================================
    # Progress cho phép tiếp tục xử lý từ vị trí đã dừng
    progress = load_progress()
    start_index = progress["last_processed_index"]  # Chỉ số query bắt đầu
    results = progress["results"]  # Kết quả đã xử lý
    
    # Tải kết quả tìm kiếm đã lưu (để tránh mất dữ liệu)
    search_results_data = load_search_results()
    
    print(f"\nTiếp tục từ query thứ {start_index + 1} (đã xử lý {len(results)} queries)")
    
    # ========================================================================
    # BƯỚC 3: XÁC ĐỊNH BATCH QUERIES CẦN XỬ LÝ
    # ========================================================================
    # Xử lý theo batch để tránh xử lý quá nhiều một lúc
    end_index = min(start_index + BATCH_SIZE, len(all_queries))  # Chỉ số query kết thúc
    queries_to_process = all_queries[start_index:end_index]  # Danh sách queries trong batch
    
    print(f"Sẽ xử lý {len(queries_to_process)} queries (từ {start_index + 1} đến {end_index})")
    print(f"Bạn có thể nhập 'exit' hoặc 'quit' bất cứ lúc nào để dừng và lưu progress\n")
    
    # ========================================================================
    # BƯỚC 4: XỬ LÝ TỪNG QUERY TRONG BATCH
    # ========================================================================
    for idx, query_info in enumerate(queries_to_process, start=start_index):
        # Lấy nội dung query
        query_text = query_info["query_text"]
        
        # Bỏ qua query nếu rỗng
        if not query_text.strip():
            print(f"\nQuery {idx + 1} bỏ qua (query_text rỗng)")
            continue
        
        print(f"\n[{idx + 1}/{len(all_queries)}] Đang xử lý query...")
        
        # Tìm kiếm top 5 kết quả phù hợp nhất với query
        # Sử dụng semantic search với ChromaDB
        search_results = search_top5(query_text)
        
        # ====================================================================
        # LƯU KẾT QUẢ TÌM KIẾM VÀO FILE
        # ====================================================================
        # Tạo entry chứa thông tin query và kết quả tìm kiếm
        search_result_entry = {
            "query_id": query_info["query_id"],  # ID của query
            "query_text": query_text,  # Nội dung query
            "category": query_info["category"],  # Phân loại
            "target_person_id": query_info["target_person_id"],  # ID hồ sơ đúng (nếu có)
            "difficulty": query_info["difficulty"],  # Độ khó
            "search_results": search_results,  # Top 5 kết quả tìm kiếm
            "timestamp": None  # Có thể thêm timestamp nếu cần
        }
        
        # Kiểm tra xem query_id đã tồn tại chưa (tránh trùng lặp khi tiếp tục)
        # Nếu đã tồn tại, cập nhật entry cũ; nếu chưa, thêm mới
        existing_idx = next((i for i, item in enumerate(search_results_data) 
                            if item.get("query_id") == query_info["query_id"]), None)
        if existing_idx is not None:
            # Cập nhật entry đã tồn tại
            search_results_data[existing_idx] = search_result_entry
        else:
            # Thêm entry mới
            search_results_data.append(search_result_entry)
        
        # Lưu vào file
        save_search_results(search_results_data)
        
        # ====================================================================
        # XỬ LÝ KẾT QUẢ TÌM KIẾM
        # ====================================================================
        if not search_results:
            # Trường hợp không tìm thấy kết quả nào
            print("Không tìm thấy kết quả nào!")
            # Metrics mặc định: tất cả đều 0
            metrics = {
                'precision_at_k': 0.0,
                'ap_at_k': 0.0,
                'relevance_labels': [0, 0, 0, 0, 0],  # Không có kết quả nào relevant
                'num_relevant': 0
            }
        else:
            # ================================================================
            # HIỂN THỊ KẾT QUẢ CHO NGƯỜI DÙNG
            # ================================================================
            # Hiển thị thông tin query và top 5 kết quả
            # Truyền query_text để tính relevance score nếu cần
            display_results(search_results, query_info, query_text)
            
            # ================================================================
            # TÍNH METRICS ĐÁNH GIÁ
            # ================================================================
            # Xác định threshold và mô tả phương pháp đánh giá
            if EVALUATION_METHOD == "distance":
                threshold = DISTANCE_THRESHOLD  # Ngưỡng distance
                method_desc = f"distance < {threshold}"  # Mô tả
            else:
                threshold = RELEVANCE_THRESHOLD  # Ngưỡng relevance score
                method_desc = f"relevance score >= {threshold}"  # Mô tả
            
            # Tính các metrics: Precision@5 và AP@5
            metrics = calculate_metrics(
                search_results,  # Kết quả tìm kiếm
                query=query_text,  # Query text
                k=5,  # Đánh giá top 5
                method=EVALUATION_METHOD,  # Phương pháp đánh giá
                threshold=threshold  # Ngưỡng
            )
            
            # Hiển thị metrics
            print("\n" + "="*80)
            print(f"METRICS ĐÁNH GIÁ (dựa trên {EVALUATION_METHOD}, {method_desc})")
            print("="*80)
            print(f"Precision@5: {metrics['precision_at_k']:.4f} ({metrics['num_relevant']}/5 relevant)")
            print(f"AP@5 (Average Precision@5): {metrics['ap_at_k']:.4f}")
            print(f"Relevance labels: {metrics['relevance_labels']}")
            
            print(f"\n💡 CÁC METRICS ĐƯỢC TÍNH DỰA TRÊN:")
            print(f"   1. Precision@K: Tỷ lệ kết quả relevant trong top K")
            print(f"   2. AP@K (Average Precision@K): Trung bình precision tại các vị trí có kết quả relevant")
            print(f"   3. MAP@K (Mean Average Precision@K): Trung bình của tất cả AP@K qua tất cả queries")
            print(f"   (MAP@K sẽ được hiển thị khi hoàn thành tất cả queries)")
            
            if EVALUATION_METHOD == "distance":
                print(f"\n📊 Tiêu chí xác định relevant: distance < {DISTANCE_THRESHOLD}")
                print(f"   (Distance càng nhỏ → càng giống → càng đúng)")
            else:
                print(f"\n📊 Tiêu chí xác định relevant: relevance score >= {RELEVANCE_THRESHOLD}")
                print(f"   (Relevance score = distance 40% + keyword matching 60%)")
            
            # Đánh giá (tự động hoặc thủ công) - giữ lại để tương thích (không dùng cho metrics)
            # Hàm này cho phép người dùng xác nhận hoặc chỉnh sửa kết quả tự động
            correct_count = get_correct_count(query_text, search_results, auto_mode=AUTO_EVALUATION)
            
            # Nếu người dùng nhập 'exit' hoặc 'quit', dừng và lưu progress
            if correct_count == -1:
                print("\nĐã dừng. Đang lưu progress...")
                progress["last_processed_index"] = idx  # Lưu chỉ số query hiện tại
                progress["results"] = results  # Lưu kết quả đã xử lý
                save_progress(progress)  # Lưu vào file
                print(f"Đã lưu progress. Đã xử lý {len(results)} queries.")
                print(f"Đã lưu kết quả tìm kiếm vào: {SEARCH_RESULTS_FILE}")
                return  # Thoát hàm
        
        # ====================================================================
        # LƯU KẾT QUẢ VÀ PROGRESS
        # ====================================================================
        # Tạo entry chứa tất cả thông tin về query và metrics
        result_entry = {
            "query_id": query_info["query_id"],  # ID query
            "query_text": query_text,  # Nội dung query
            "category": query_info["category"],  # Phân loại
            "target_person_id": query_info["target_person_id"],  # ID hồ sơ đúng
            "difficulty": query_info["difficulty"],  # Độ khó
            "precision_at_5": metrics['precision_at_k'],  # Precision@5
            "ap_at_5": metrics['ap_at_k'],  # AP@5
            "relevance_labels": metrics['relevance_labels'],  # Nhãn của từng kết quả
            "num_relevant": metrics['num_relevant'],  # Số lượng relevant
            "search_results": search_results  # Kết quả tìm kiếm chi tiết
        }
        
        # Thêm vào danh sách kết quả
        results.append(result_entry)
        
        # Cập nhật và lưu progress sau mỗi query
        # Điều này đảm bảo không mất dữ liệu nếu script bị dừng đột ngột
        progress["last_processed_index"] = idx + 1  # Cập nhật chỉ số query đã xử lý
        progress["results"] = results  # Cập nhật kết quả
        save_progress(progress)  # Lưu vào file
        
        # Thông báo đã lưu
        print(f"✓ Đã lưu. Precision@5: {metrics['precision_at_k']:.4f} ({metrics['num_relevant']}/5)")
        print(f"✓ Đã lưu kết quả tìm kiếm vào file: {SEARCH_RESULTS_FILE.name}")
    
    # ========================================================================
    # KIỂM TRA VÀ TÍNH THỐNG KÊ TỔNG HỢP
    # ========================================================================
    # Kiểm tra xem đã xử lý hết tất cả queries chưa
    if end_index >= len(all_queries):
        print("\n" + "="*80)
        print("ĐÃ XỬ LÝ HẾT TẤT CẢ QUERIES!")
        print("="*80)
        
        # ====================================================================
        # TÍNH THỐNG KÊ TỔNG HỢP
        # ====================================================================
        total = len(results)  # Tổng số queries đã xử lý
        if total > 0:
            # Tính MAP@5 (Mean Average Precision@5)
            # MAP@5 = trung bình của tất cả AP@5 qua tất cả queries
            # Đây là metric quan trọng nhất để đánh giá chất lượng hệ thống
            ap_scores = [r.get("ap_at_5", 0.0) for r in results]  # Lấy tất cả AP@5
            map_at_5 = sum(ap_scores) / total if total > 0 else 0.0  # Tính trung bình
            
            # Tính Precision@5 trung bình
            # Trung bình của tất cả Precision@5 qua tất cả queries
            precision_scores = [r.get("precision_at_5", 0.0) for r in results]  # Lấy tất cả Precision@5
            avg_precision_at_5 = sum(precision_scores) / total if total > 0 else 0.0  # Tính trung bình
            
            # Phân tích phân phối Precision@5
            precision_sorted = sorted(precision_scores)
            min_precision = min(precision_scores)
            max_precision = max(precision_scores)
            median_precision = precision_sorted[total // 2] if total > 0 else 0.0
            
            # Đếm số queries theo mức Precision@5
            perfect_queries = sum(1 for p in precision_scores if p == 1.0)
            high_queries = sum(1 for p in precision_scores if 0.8 <= p < 1.0)
            medium_queries = sum(1 for p in precision_scores if 0.5 <= p < 0.8)
            low_queries = sum(1 for p in precision_scores if p < 0.5)
            
            # Phân tích phân phối AP@5
            ap_sorted = sorted(ap_scores)
            min_ap = min(ap_scores)
            max_ap = max(ap_scores)
            median_ap = ap_sorted[total // 2] if total > 0 else 0.0
            
            # Phân tích số lượng relevant results
            all_num_relevant = [r.get("num_relevant", 0) for r in results]
            avg_relevant = sum(all_num_relevant) / total if total > 0 else 0.0
            total_relevant_all = sum(all_num_relevant)
            max_possible = total * 5  # Mỗi query có 5 kết quả
            
            # Phân tích distance (nếu có)
            all_distances = []
            for r in results:
                for sr in r.get("search_results", []):
                    dist = sr.get("distance")
                    if dist is not None:
                        all_distances.append(dist)
            
            print(f"\nTổng số queries đã xử lý: {total}")
            print(f"\n{'='*80}")
            print("METRICS TỔNG KẾT")
            print(f"{'='*80}")
            print(f"Precision@5 trung bình: {avg_precision_at_5:.4f}")
            print(f"MAP@5 (Mean Average Precision@5): {map_at_5:.4f}")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH CHI TIẾT PRECISION@5")
            print(f"{'='*80}")
            print(f"Min: {min_precision:.4f} | Max: {max_precision:.4f} | Median: {median_precision:.4f}")
            print(f"\nPhân bố Precision@5:")
            print(f"  Perfect (1.0000): {perfect_queries} queries ({perfect_queries/total*100:.1f}%)")
            print(f"  High (0.80-0.99): {high_queries} queries ({high_queries/total*100:.1f}%)")
            print(f"  Medium (0.50-0.79): {medium_queries} queries ({medium_queries/total*100:.1f}%)")
            print(f"  Low (<0.50): {low_queries} queries ({low_queries/total*100:.1f}%)")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH CHI TIẾT AP@5")
            print(f"{'='*80}")
            print(f"Min: {min_ap:.4f} | Max: {max_ap:.4f} | Median: {median_ap:.4f}")
            
            print(f"\n{'='*80}")
            print("PHÂN TÍCH SỐ LƯỢNG RELEVANT RESULTS")
            print(f"{'='*80}")
            print(f"Tổng số kết quả relevant: {total_relevant_all}/{max_possible}")
            print(f"Tỷ lệ relevant: {total_relevant_all/max_possible*100:.2f}%")
            print(f"Số lượng relevant trung bình mỗi query: {avg_relevant:.2f}/5")
            
            if all_distances:
                avg_distance = sum(all_distances) / len(all_distances)
                min_distance = min(all_distances)
                max_distance = max(all_distances)
                distance_sorted = sorted(all_distances)
                median_distance = distance_sorted[len(distance_sorted) // 2]
                
                print(f"\n{'='*80}")
                print("PHÂN TÍCH DISTANCE")
                print(f"{'='*80}")
                print(f"Distance trung bình: {avg_distance:.4f}")
                print(f"Min: {min_distance:.4f} | Max: {max_distance:.4f} | Median: {median_distance:.4f}")
                if EVALUATION_METHOD == "distance":
                    relevant_distances = [d for d in all_distances if d < DISTANCE_THRESHOLD]
                    non_relevant_distances = [d for d in all_distances if d >= DISTANCE_THRESHOLD]
                    print(f"\nVới threshold = {DISTANCE_THRESHOLD}:")
                    print(f"  Relevant: {len(relevant_distances)} kết quả ({len(relevant_distances)/len(all_distances)*100:.1f}%)")
                    print(f"  Non-relevant: {len(non_relevant_distances)} kết quả ({len(non_relevant_distances)/len(all_distances)*100:.1f}%)")
                    if relevant_distances:
                        print(f"  Distance trung bình của relevant: {sum(relevant_distances)/len(relevant_distances):.4f}")
                    if non_relevant_distances:
                        print(f"  Distance trung bình của non-relevant: {sum(non_relevant_distances)/len(non_relevant_distances):.4f}")
            
            # Thống kê theo category
            category_stats = {}
            for r in results:
                cat = r["category"]
                if cat not in category_stats:
                    category_stats[cat] = {
                        "count": 0, 
                        "total_precision": 0,
                        "total_ap": 0
                    }
                category_stats[cat]["count"] += 1
                category_stats[cat]["total_precision"] += r.get("precision_at_5", 0.0)
                category_stats[cat]["total_ap"] += r.get("ap_at_5", 0.0)
            
            # Top queries tốt nhất và xấu nhất
            results_with_scores = [(r, r.get("precision_at_5", 0.0), r.get("ap_at_5", 0.0)) for r in results]
            results_with_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            print(f"\n{'='*80}")
            print("TOP 5 QUERIES TỐT NHẤT (theo Precision@5)")
            print(f"{'='*80}")
            for i, (r, prec, ap) in enumerate(results_with_scores[:5], 1):
                print(f"{i}. Query ID: {r['query_id']} | Category: {r['category']} | Difficulty: {r['difficulty']}")
                print(f"   Precision@5: {prec:.4f} | AP@5: {ap:.4f} | Relevant: {r.get('num_relevant', 0)}/5")
                print(f"   Query: {r['query_text'][:80]}...")
            
            print(f"\n{'='*80}")
            print("TOP 5 QUERIES XẤU NHẤT (theo Precision@5)")
            print(f"{'='*80}")
            for i, (r, prec, ap) in enumerate(results_with_scores[-5:], 1):
                print(f"{i}. Query ID: {r['query_id']} | Category: {r['category']} | Difficulty: {r['difficulty']}")
                print(f"   Precision@5: {prec:.4f} | AP@5: {ap:.4f} | Relevant: {r.get('num_relevant', 0)}/5")
                print(f"   Query: {r['query_text'][:80]}...")
            
            print(f"\n{'='*80}")
            print("THỐNG KÊ THEO CATEGORY")
            print(f"{'='*80}")
            for cat, stats in sorted(category_stats.items()):
                avg_prec = stats["total_precision"] / stats["count"]
                avg_ap = stats["total_ap"] / stats["count"]
                # Tính min, max cho category này
                cat_precisions = [r.get("precision_at_5", 0.0) for r in results if r["category"] == cat]
                cat_min = min(cat_precisions) if cat_precisions else 0.0
                cat_max = max(cat_precisions) if cat_precisions else 0.0
                cat_perfect = sum(1 for p in cat_precisions if p == 1.0)
                print(f"  {cat} (n={stats['count']}):")
                print(f"    Precision@5: {avg_prec:.4f} (min: {cat_min:.4f}, max: {cat_max:.4f}, perfect: {cat_perfect})")
                print(f"    AP@5: {avg_ap:.4f}")
            
            # Thống kê theo difficulty
            difficulty_stats = {}
            for r in results:
                diff = r["difficulty"]
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {
                        "count": 0, 
                        "total_precision": 0,
                        "total_ap": 0
                    }
                difficulty_stats[diff]["count"] += 1
                difficulty_stats[diff]["total_precision"] += r.get("precision_at_5", 0.0)
                difficulty_stats[diff]["total_ap"] += r.get("ap_at_5", 0.0)
            
            print(f"\n{'='*80}")
            print("THỐNG KÊ THEO DIFFICULTY")
            print(f"{'='*80}")
            for diff, stats in sorted(difficulty_stats.items()):
                avg_prec = stats["total_precision"] / stats["count"]
                avg_ap = stats["total_ap"] / stats["count"]
                # Tính min, max cho difficulty này
                diff_precisions = [r.get("precision_at_5", 0.0) for r in results if r["difficulty"] == diff]
                diff_min = min(diff_precisions) if diff_precisions else 0.0
                diff_max = max(diff_precisions) if diff_precisions else 0.0
                diff_perfect = sum(1 for p in diff_precisions if p == 1.0)
                print(f"  {diff} (n={stats['count']}):")
                print(f"    Precision@5: {avg_prec:.4f} (min: {diff_min:.4f}, max: {diff_max:.4f}, perfect: {diff_perfect})")
                print(f"    AP@5: {avg_ap:.4f}")
            
            # Phân tích và đề xuất threshold
            if EVALUATION_METHOD == "distance" and all_distances:
                print(f"\n{'='*80}")
                print("PHÂN TÍCH VÀ ĐỀ XUẤT THRESHOLD")
                print(f"{'='*80}")
                print(f"Threshold hiện tại: {DISTANCE_THRESHOLD}")
                print(f"\nPhân tích với các threshold khác nhau:")
                test_thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                for thresh in test_thresholds:
                    relevant_count = sum(1 for d in all_distances if d < thresh)
                    relevant_pct = relevant_count / len(all_distances) * 100
                    print(f"  Threshold {thresh:.2f}: {relevant_count}/{len(all_distances)} relevant ({relevant_pct:.1f}%)")
                
                # Đề xuất threshold dựa trên phân vị
                if len(all_distances) >= 10:
                    p25 = distance_sorted[len(distance_sorted) // 4]
                    p50 = median_distance
                    p75 = distance_sorted[len(distance_sorted) * 3 // 4]
                    print(f"\nPhân vị distance:")
                    print(f"  25th percentile (P25): {p25:.4f}")
                    print(f"  50th percentile (Median): {p50:.4f}")
                    print(f"  75th percentile (P75): {p75:.4f}")
                    print(f"\n💡 Đề xuất:")
                    print(f"  - Threshold chặt chẽ (P25): {p25:.4f} → ~{sum(1 for d in all_distances if d < p25)/len(all_distances)*100:.1f}% relevant")
                    print(f"  - Threshold vừa phải (P50): {p50:.4f} → ~{sum(1 for d in all_distances if d < p50)/len(all_distances)*100:.1f}% relevant")
                    print(f"  - Threshold lỏng (P75): {p75:.4f} → ~{sum(1 for d in all_distances if d < p75)/len(all_distances)*100:.1f}% relevant")
            
            # Tóm tắt cuối cùng
            print(f"\n{'='*80}")
            print("TÓM TẮT ĐÁNH GIÁ")
            print(f"{'='*80}")
            print(f"📊 Tổng quan:")
            print(f"   - Tổng số queries: {total}")
            print(f"   - Precision@5 trung bình: {avg_precision_at_5:.4f} ({avg_precision_at_5*100:.2f}%)")
            print(f"   - MAP@5: {map_at_5:.4f} ({map_at_5*100:.2f}%)")
            print(f"   - Số queries đạt perfect (1.0): {perfect_queries}/{total} ({perfect_queries/total*100:.1f}%)")
            print(f"   - Số queries có Precision@5 >= 0.8: {perfect_queries + high_queries}/{total} ({(perfect_queries + high_queries)/total*100:.1f}%)")
            print(f"\n📈 Chất lượng:")
            if avg_precision_at_5 >= 0.9:
                print(f"   ✓ Hệ thống hoạt động RẤT TỐT (Precision@5 >= 90%)")
            elif avg_precision_at_5 >= 0.8:
                print(f"   ✓ Hệ thống hoạt động TỐT (Precision@5 >= 80%)")
            elif avg_precision_at_5 >= 0.7:
                print(f"   ⚠ Hệ thống hoạt động KHÁ (Precision@5 >= 70%)")
            else:
                print(f"   ⚠ Hệ thống cần CẢI THIỆN (Precision@5 < 70%)")
            
            if EVALUATION_METHOD == "distance":
                print(f"\n⚙️  Cấu hình đánh giá:")
                print(f"   - Phương pháp: Distance-based")
                print(f"   - Threshold: {DISTANCE_THRESHOLD}")
                print(f"   - Tiêu chí: distance < {DISTANCE_THRESHOLD} → relevant")
            else:
                print(f"\n⚙️  Cấu hình đánh giá:")
                print(f"   - Phương pháp: Relevance score-based")
                print(f"   - Threshold: {RELEVANCE_THRESHOLD}")
                print(f"   - Tiêu chí: relevance score >= {RELEVANCE_THRESHOLD} → relevant")
        
        # ====================================================================
        # LƯU KẾT QUẢ CUỐI CÙNG
        # ====================================================================
        # Lưu tất cả kết quả vào file JSON
        save_results(results)
        print(f"\nĐã lưu kết quả vào: {RESULTS_FILE}")
        print(f"Đã lưu kết quả tìm kiếm để đánh giá vào: {SEARCH_RESULTS_FILE}")
        
        # Xóa file progress vì đã hoàn thành tất cả queries
        # File progress chỉ cần khi đang xử lý dở dang
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()  # Xóa file
            print("Đã xóa file progress.")
    else:
        # ====================================================================
        # CHƯA XỬ LÝ HẾT - THÔNG BÁO TIẾP TỤC
        # ====================================================================
        # Nếu chưa xử lý hết, thông báo số queries đã xử lý và còn lại
        print(f"\nĐã xử lý {len(queries_to_process)} queries trong batch này.")
        print(f"Còn lại {len(all_queries) - end_index} queries.")
        print(f"Chạy lại script để tiếp tục từ query {end_index + 1}")


# ============================================================================
# ENTRY POINT - ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ============================================================================
if __name__ == "__main__":
    """
    Điểm bắt đầu của chương trình.
    
    Khi chạy script trực tiếp (không import), hàm process_queries() sẽ được gọi.
    Có xử lý exception để:
    - Bắt KeyboardInterrupt (Ctrl+C) và thông báo progress đã được lưu
    - Bắt các exception khác và hiển thị lỗi chi tiết
    """
    try:
        # Gọi hàm chính để xử lý queries
        process_queries()
    except KeyboardInterrupt:
        # Người dùng nhấn Ctrl+C để dừng
        # Progress đã được lưu tự động trong process_queries()
        print("\n\nĐã dừng bởi người dùng (Ctrl+C). Progress đã được lưu.")
    except Exception as e:
        # Xử lý các lỗi khác
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()  # In stack trace để debug

