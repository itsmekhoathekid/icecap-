import argparse
import json
import numpy as np
import fasttext
import tqdm

def main(params):
    input_json_path = f"/data/npl/ICEK/ICECAP/icecap-/{params['dataset']}_data/{params['dataset']}_cap_basic.json"
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

    output_path = f"/data/npl/ICEK/ICECAP/icecap-/{params['dataset']}_data/{params['dataset']}_vocab_emb.npy"
    np.save(output_path, embedding_matrix)
    print('Ma trận nhúng đã được lưu tại', output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ViWiki', choices=['breakingnews', 'goodnews', 'ViWiki'])
    parser.add_argument('--emb_path', default='/data/npl/ICEK/ICECAP/icecap-/data/cc.vi.300.bin', help='Đường dẫn đến tệp nhúng')
    args = parser.parse_args()
    params = vars(args)
    print('Các tham số đầu vào đã phân tích:')
    print(json.dumps(params, indent=2, ensure_ascii=False))
    main(params)
# import argparse
# import json
# import numpy as np
# from gensim.models import KeyedVectors
# import tqdm

# def main(params):
#     input_json_path = f"/data/npl/ICEK/ICECAP/icecap-/{params['dataset']}_data/{params['dataset']}_cap_basic.json"
#     print('Đang tải tệp JSON:', input_json_path)
#     with open(input_json_path, 'r', encoding='utf-8') as f:
#         info = json.load(f)
#     ix_to_word = info['ix_to_word']
#     vocab_size = len(ix_to_word)
#     print('Kích thước từ vựng là', vocab_size)

#     # Load word2vec text model (PhoW2V, etc.)
#     print('Đang tải embedding từ', params['emb_path'])
#     model = KeyedVectors.load_word2vec_format(params['emb_path'], binary=False)

#     embedding_dim = model.vector_size
#     embedding_matrix = np.zeros((vocab_size + 1, embedding_dim), dtype=np.float32)

#     non_found_num = 0
#     for index, word in tqdm.tqdm(ix_to_word.items(), desc = 'Đang process data'):
#         if word in model:
#             embedding_matrix[int(index)] = model[word]
#         elif word.lower() in model:
#             embedding_matrix[int(index)] = model[word.lower()]
#         else:
#             non_found_num += 1
#             embedding_matrix[int(index)] = np.random.uniform(-0.1, 0.1, embedding_dim)
    
#     print(f"{non_found_num}/{vocab_size} từ không tìm thấy trong embedding")
#     print('Kích thước ma trận embedding:', embedding_matrix.shape)

#     output_path = f"/data/npl/ICEK/ICECAP/icecap-/{params['dataset']}_data/{params['dataset']}_vocab_emb.npy"
#     np.save(output_path, embedding_matrix)
#     print('Ma trận embedding đã được lưu tại', output_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='breakingnews', choices=['breakingnews', 'goodnews', 'ViWiki'])
#     parser.add_argument('--emb_path', default='/data/npl/ICEK/ICECAP/icecap-/data/word2vec_vi_syllables_300dims.txt',
#                         help='Đường dẫn đến file word2vec text embedding')
#     args = parser.parse_args()
#     params = vars(args)
#     print('Các tham số đầu vào đã phân tích:')
#     print(json.dumps(params, indent=2, ensure_ascii=False))
#     main(params)
