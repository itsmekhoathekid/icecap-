import json

# Hàm load file JSON
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None

# Hàm in 200 dòng đầu tiên
def print_first_200_lines(data):
    try:
        # Nếu là danh sách
        if isinstance(data, list):
            for i, item in enumerate(data[:200]):
                print(f"Row {i+1}: {item}")
        
        # Nếu là dictionary
        elif isinstance(data, dict):
            for i, (key, value) in enumerate(list(data.items())[:200]):
                print(f"Key {i+1}: {key} -> {value}")
        else:
            print("Dữ liệu không phải là list hoặc dictionary.")
    except Exception as e:
        print(f"Lỗi khi in dữ liệu: {e}")

# Đường dẫn đến các file JSON
file_paths = [
    # r"C:\Users\VIET HOANG - VTS\Downloads\goodnews_data\goodnews_data\goodnews_ttv.json"
    r"C:\Users\VIET HOANG - VTS\Downloads\captioning_dataset.json"
]

# Load và in 200 dòng đầu tiên từ từng file
for file_path in file_paths:
    print(f"\nĐang xử lý file: {file_path}")
    data = load_json(file_path)
    if data:
        print_first_200_lines(data)


# article_ner : related sentences
# ner : dictionary chứa {key (tên entity): value (loại entity-stt)}