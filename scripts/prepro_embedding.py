import argparse
import json
import numpy as np
import fasttext
import tqdm

def main(params):
    input_json_path = f"/content/ICECAP/{params['dataset']}_data/{params['dataset']}_cap_basic.json"
    print('Đang tải tệp JSON:', input_json_path)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    ix_to_word = info['ix_to_word']
    vocab_size = len(ix_to_word)
    print('Kích thước từ vựng là', vocab_size)

    print('Đang tải mô hình fastText từ', params['emb_path'])
    model = fasttext.load_model(params['emb_path'])

    embedding_dim = model.get_dimension()
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim), dtype=np.float32)

    non_found_num = 0
    for index, word in tqdm.tqdm(ix_to_word.items(), desc = 'Đang process data'):
        if word in model:
            embedding_matrix[int(index)] = model.get_word_vector(word)
        elif word.lower() in model:
            embedding_matrix[int(index)] = model.get_word_vector(word.lower())
        else:
            non_found_num += 1
            embedding_matrix[int(index)] = np.random.uniform(-0.1, 0.1, embedding_dim)
    
    print(f"{non_found_num}/{vocab_size} từ không tìm thấy trong tệp nhúng")
    print('Kích thước ma trận nhúng', embedding_matrix.shape)

    output_path = f"/content/ICECAP/{params['dataset']}_data/{params['dataset']}_vocab_emb.npy"
    np.save(output_path, embedding_matrix)
    print('Ma trận nhúng đã được lưu tại', output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breakingnews', choices=['breakingnews', 'goodnews', 'ViWiki'])
    parser.add_argument('--emb_path', default='/content/cc.vi.300.bin', help='Đường dẫn đến tệp nhúng')
    args = parser.parse_args()
    params = vars(args)
    print('Các tham số đầu vào đã phân tích:')
    print(json.dumps(params, indent=2, ensure_ascii=False))
    main(params)
