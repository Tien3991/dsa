import re
from collections import Counter
import string

## --------------------------------------------------
## PHẦN 1: CẤU TRÚC DỮ LIỆU CƠ BẢN
## --------------------------------------------------

class Node:
    """
    Node đại diện cho một phần tử trong danh sách liên kết.
    Lưu chỉ số của từ trong từ điển.
    """
    def __init__(self, index):
        self.index = index  # Chỉ số của từ trong từ điển
        self.next = None    # Con trỏ đến node tiếp theo

class SparseVector:
    """
    Vector thưa được biểu diễn bằng danh sách liên kết.
    Chỉ lưu các chỉ số có giá trị 1 (từ xuất hiện trong văn bản).
    """
    def __init__(self):
        self.head = None
        self.length = 0  # Số lượng từ khác 0 trong vector
        
    def insert(self, index):
        """
        Chèn chỉ số vào vector theo thứ tự tăng dần.
        Nếu chỉ số đã tồn tại thì bỏ qua.
        """
        new_node = Node(index)
        
        # Danh sách rỗng
        if self.head is None:
            self.head = new_node
            self.length += 1
            return
        
        # Chèn vào đầu
        if index < self.head.index:
            new_node.next = self.head
            self.head = new_node
            self.length += 1
            return
        
        # Kiểm tra trùng ở đầu
        if index == self.head.index:
            return
            
        # Tìm vị trí chèn
        current = self.head
        while current.next and current.next.index < index:
            current = current.next
        
        # Kiểm tra trùng
        if current.next and current.next.index == index:
            return
            
        # Chèn vào giữa hoặc cuối
        new_node.next = current.next
        current.next = new_node
        self.length += 1
    
    def get_indices(self):
        """Trả về danh sách các chỉ số trong vector."""
        indices = []
        current = self.head
        while current:
            indices.append(current.index)
            current = current.next
        return indices
    
    def print_vector(self):
        """In vector dưới dạng danh sách liên kết."""
        if self.head is None:
            print("Empty vector")
            return
        nodes = []
        current = self.head
        while current:
            nodes.append(str(current.index))
            current = current.next
        print(" -> ".join(nodes) + " -> None")
    
    def to_binary_string(self, vocab_size):
        """Chuyển vector thành chuỗi nhị phân đầy đủ."""
        binary = ['0'] * vocab_size
        current = self.head
        while current:
            if current.index < vocab_size:
                binary[current.index] = '1'
            current = current.next
        return ''.join(binary)

## --------------------------------------------------
## PHẦN 2: XỬ LÝ VĂN BẢN VÀ TỪ ĐIỂN
## --------------------------------------------------

class TextProcessor:
    """
    Lớp xử lý văn bản và quản lý từ điển.
    """
    def __init__(self):
        self.vocabulary = {}  # Từ -> Chỉ số
        self.inverse_vocab = {}  # Chỉ số -> Từ
        self.vocab_size = 0
        self.documents = []  # Lưu các văn bản gốc
        
    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản: chuyển thường, loại bỏ dấu câu, tách từ.
        """
        # Chuyển thường
        text = text.lower()
        # Loại bỏ dấu câu
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tách từ
        words = text.split()
        return words
    
    def build_vocabulary(self, documents):
        """
        Xây dựng từ điển từ tập văn bản.
        """
        all_words = set()
        for doc in documents:
            words = self.preprocess_text(doc)
            all_words.update(words)
        
        # Sắp xếp từ và gán chỉ số
        sorted_words = sorted(list(all_words))
        for idx, word in enumerate(sorted_words):
            self.vocabulary[word] = idx
            self.inverse_vocab[idx] = word
        
        self.vocab_size = len(self.vocabulary)
        self.documents = documents
        print(f"Từ điển được xây dựng với {self.vocab_size} từ")
        
    def text_to_sparse_vector(self, text):
        """
        Chuyển văn bản thành vector thưa.
        """
        words = self.preprocess_text(text)
        vector = SparseVector()
        
        # Chỉ thêm các từ có trong từ điển
        added_indices = set()
        for word in words:
            if word in self.vocabulary and self.vocabulary[word] not in added_indices:
                vector.insert(self.vocabulary[word])
                added_indices.add(self.vocabulary[word])
        
        return vector
    
    def print_vocabulary(self, limit=20):
        """In một phần của từ điển."""
        print(f"\nTừ điển (hiển thị {min(limit, self.vocab_size)} từ đầu tiên):")
        count = 0
        for word, idx in sorted(self.vocabulary.items(), key=lambda x: x[1]):
            print(f"  {idx}: '{word}'")
            count += 1
            if count >= limit:
                break

## --------------------------------------------------
## PHẦN 3: CÁC PHÉP TOÁN TRÊN VECTOR THƯA
## --------------------------------------------------

def merge_vectors(vec1, vec2):
    """
    Gộp (hợp) hai vector thưa - phép cộng vector.
    Kết quả chứa tất cả các từ từ cả hai văn bản.
    """
    result = SparseVector()
    
    p1 = vec1.head
    p2 = vec2.head
    
    # Dùng dummy head để đơn giản hóa việc xây dựng danh sách mới
    dummy = Node(-1)
    tail = dummy
    
    while p1 or p2:
        if p1 is None:
            tail.next = Node(p2.index)
            p2 = p2.next
        elif p2 is None:
            tail.next = Node(p1.index)
            p1 = p1.next
        elif p1.index < p2.index:
            tail.next = Node(p1.index)
            p1 = p1.next
        elif p2.index < p1.index:
            tail.next = Node(p2.index)
            p2 = p2.next
        else:  # p1.index == p2.index
            tail.next = Node(p1.index)
            p1 = p1.next
            p2 = p2.next
        
        tail = tail.next
        result.length += 1
    
    result.head = dummy.next
    return result

def intersection_vectors(vec1, vec2):
    """
    Tìm giao của hai vector thưa.
    Kết quả chứa các từ xuất hiện trong cả hai văn bản.
    """
    result = SparseVector()
    
    p1 = vec1.head
    p2 = vec2.head
    
    dummy = Node(-1)
    tail = dummy
    
    while p1 and p2:
        if p1.index < p2.index:
            p1 = p1.next
        elif p2.index < p1.index:
            p2 = p2.next
        else:  # p1.index == p2.index
            tail.next = Node(p1.index)
            tail = tail.next
            result.length += 1
            p1 = p1.next
            p2 = p2.next
    
    result.head = dummy.next
    return result

def hamming_distance(vec1, vec2):
    """
    Tính khoảng cách Hamming giữa hai vector thưa.
    Đếm số vị trí mà hai vector khác nhau.
    """
    # Công thức: |A| + |B| - 2*|A ∩ B|
    intersection = intersection_vectors(vec1, vec2)
    distance = vec1.length + vec2.length - 2 * intersection.length
    return distance

def jaccard_similarity(vec1, vec2):
    """
    Tính độ tương đồng Jaccard giữa hai vector.
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    intersection = intersection_vectors(vec1, vec2)
    union = merge_vectors(vec1, vec2)
    
    if union.length == 0:
        return 0.0
    
    similarity = intersection.length / union.length
    return similarity

## --------------------------------------------------
## PHẦN 4: HỆ THỐNG SO SÁNH VĂN BẢN
## --------------------------------------------------

class DocumentSimilaritySystem:
    """
    Hệ thống so sánh độ tương đồng văn bản.
    """
    def __init__(self):
        self.processor = TextProcessor()
        self.document_vectors = []
        
    def add_documents(self, documents):
        """Thêm và xử lý các văn bản."""
        self.processor.build_vocabulary(documents)
        
        # Chuyển mỗi văn bản thành vector thưa
        for doc in documents:
            vector = self.processor.text_to_sparse_vector(doc)
            self.document_vectors.append(vector)
            
    def compare_documents(self, doc_idx1, doc_idx2):
        """So sánh hai văn bản theo chỉ số."""
        if doc_idx1 >= len(self.document_vectors) or doc_idx2 >= len(self.document_vectors):
            print("Chỉ số văn bản không hợp lệ!")
            return
        
        vec1 = self.document_vectors[doc_idx1]
        vec2 = self.document_vectors[doc_idx2]
        
        print(f"\n=== So sánh Văn bản {doc_idx1} và Văn bản {doc_idx2} ===")
        print(f"Văn bản {doc_idx1}: {self.processor.documents[doc_idx1][:50]}...")
        print(f"Văn bản {doc_idx2}: {self.processor.documents[doc_idx2][:50]}...")
        
        # Tính các độ đo
        hamming = hamming_distance(vec1, vec2)
        jaccard = jaccard_similarity(vec1, vec2)
        intersection = intersection_vectors(vec1, vec2)
        
        print(f"\nKết quả:")
        print(f"  - Số từ trong văn bản {doc_idx1}: {vec1.length}")
        print(f"  - Số từ trong văn bản {doc_idx2}: {vec2.length}")
        print(f"  - Số từ chung (giao): {intersection.length}")
        print(f"  - Khoảng cách Hamming: {hamming}")
        print(f"  - Độ tương đồng Jaccard: {jaccard:.2%}")
        
        # Hiển thị các từ chung
        if intersection.length > 0:
            print(f"\nCác từ chung:")
            indices = intersection.get_indices()
            common_words = [self.processor.inverse_vocab[idx] for idx in indices[:10]]
            print(f"  {', '.join(common_words)}")
            if len(indices) > 10:
                print(f"  ... và {len(indices) - 10} từ khác")
    
    def find_most_similar(self, doc_idx):
        """Tìm văn bản tương tự nhất với văn bản cho trước."""
        if doc_idx >= len(self.document_vectors):
            print("Chỉ số văn bản không hợp lệ!")
            return
        
        target_vec = self.document_vectors[doc_idx]
        similarities = []
        
        for i, vec in enumerate(self.document_vectors):
            if i != doc_idx:
                sim = jaccard_similarity(target_vec, vec)
                similarities.append((i, sim))
        
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n=== Văn bản tương tự nhất với Văn bản {doc_idx} ===")
        print(f"Văn bản gốc: {self.processor.documents[doc_idx][:50]}...")
        print("\nTop 3 văn bản tương tự:")
        
        for i, (idx, sim) in enumerate(similarities[:3], 1):
            print(f"{i}. Văn bản {idx} (Độ tương đồng: {sim:.2%})")
            print(f"   {self.processor.documents[idx][:50]}...")

## --------------------------------------------------
## PHẦN 5: DEMO VÀ TEST
## --------------------------------------------------

def main():
    print("=== HỆ THỐNG PHÂN TÍCH ĐỘ TƯƠNG ĐỒNG VĂN BẢN ===\n")
    
    # Tạo các văn bản mẫu (giả lập crawl data)
    documents = [
        "Python là ngôn ngữ lập trình phổ biến cho khoa học dữ liệu và machine learning",
        "Machine learning và deep learning là các lĩnh vực quan trọng của trí tuệ nhân tạo",
        "Python cung cấp nhiều thư viện mạnh mẽ như NumPy Pandas và Scikit-learn",
        "Cấu trúc dữ liệu như danh sách liên kết cây và đồ thị rất quan trọng trong lập trình",
        "Danh sách liên kết là cấu trúc dữ liệu cơ bản trong khoa học máy tính",
        "Deep learning sử dụng mạng neural để học từ dữ liệu lớn",
        "Python và Java là hai ngôn ngữ lập trình phổ biến trong phát triển phần mềm"
    ]
    
    # Khởi tạo hệ thống
    system = DocumentSimilaritySystem()
    system.add_documents(documents)
    
    # In từ điển
    system.processor.print_vocabulary(15)
    
    print("\n" + "="*60)
    print("DEMO CÁC CHỨC NĂNG")
    print("="*60)
    
    # 1. Demo chuyển văn bản thành vector thưa
    print("\n1. CHUYỂN VĂN BẢN THÀNH VECTOR THƯA")
    print("-" * 40)
    test_text = "Python và machine learning"
    print(f"Văn bản test: '{test_text}'")
    test_vector = system.processor.text_to_sparse_vector(test_text)
    print(f"Vector thưa (các chỉ số của từ xuất hiện):")
    test_vector.print_vector()
    
    # Hiển thị các từ tương ứng
    indices = test_vector.get_indices()
    words = [system.processor.inverse_vocab[idx] for idx in indices]
    print(f"Các từ tương ứng: {words}")
    
    # 2. Demo gộp văn bản
    print("\n2. GỘP HAI VĂN BẢN (PHÉP HỢP)")
    print("-" * 40)
    vec1 = system.document_vectors[0]
    vec2 = system.document_vectors[1]
    merged = merge_vectors(vec1, vec2)
    
    print(f"Văn bản 1 có {vec1.length} từ")
    print(f"Văn bản 2 có {vec2.length} từ")
    print(f"Văn bản gộp có {merged.length} từ (loại bỏ trùng)")
    
    # 3. Demo so sánh văn bản
    print("\n3. SO SÁNH ĐỘ TƯƠNG ĐỒNG")
    print("-" * 40)
    system.compare_documents(0, 2)  # So sánh văn bản về Python
    system.compare_documents(1, 5)  # So sánh văn bản về deep learning
    system.compare_documents(3, 4)  # So sánh văn bản về cấu trúc dữ liệu
    
    # 4. Tìm văn bản tương tự nhất
    print("\n4. TÌM VĂN BẢN TƯƠNG TỰ NHẤT")
    print("-" * 40)
    system.find_most_similar(0)  # Tìm văn bản tương tự với văn bản đầu tiên
    
    # 5. Demo với văn bản người dùng nhập
    print("\n5. PHÂN TÍCH VĂN BẢN MỚI")
    print("-" * 40)
    new_text = "Lập trình Python với cấu trúc dữ liệu và thuật toán"
    print(f"Văn bản mới: '{new_text}'")
    new_vector = system.processor.text_to_sparse_vector(new_text)
    
    print(f"Số từ trong văn bản mới (có trong từ điển): {new_vector.length}")
    
    # So sánh với tất cả văn bản hiện có
    print("\nĐộ tương đồng với các văn bản trong hệ thống:")
    for i, doc_vec in enumerate(system.document_vectors):
        similarity = jaccard_similarity(new_vector, doc_vec)
        hamming = hamming_distance(new_vector, doc_vec)
        print(f"  Văn bản {i}: Jaccard = {similarity:.2%}, Hamming = {hamming}")

if __name__ == "__main__":
    main()
