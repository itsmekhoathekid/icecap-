import tqdm
import h5py
import spacy
import numpy as np
import json
from collections import Counter
from nltk.tokenize import word_tokenize
import math
import sys
import fasttext
import fasttext.util
from py_vncorenlp import VnCoreNLP

def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_h5(file_save, data):
    f_lb = h5py.File(file_save, "w")
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    ds = f_lb.create_dataset('average', (len(data), 300, ), dtype=dt)
    for i,d in tqdm.tqdm(enumerate(data)):
        ds[i] = d
    f_lb.close()


def getLog(frequency, a=10 ** -3):
    #     return 1/math.log(1.1) if frequency==1 else 1/math.log(frequency)
    return a / (a + math.log(1 + frequency))

def get_embed(token):
    """
    Lấy vector embedding cho một token.
    
    Args:
    - model: Mô hình embedding (fastText, GloVe, Word2Vec, etc.)
    - token: Từ cần lấy vector.
    
    Returns:
    - vector: Vector embedding của từ hoặc None nếu không có.
    """
    try:
        return model_embed[token]
    except KeyError:
        return None

def get_vector_avg_weighted_full(sent):
    """
    Tính vector trung bình có trọng số cho một câu dựa trên embedding.
    
    Args:
    - model: Mô hình embedding (fastText, GloVe, Word2Vec, etc.)
    - sent: Câu đầu vào (list các từ hoặc chuỗi).
    
    Returns:
    - doc_vector: Vector đại diện cho câu.
    """
    # Nếu sent là chuỗi, chuyển thành danh sách từ
    if isinstance(sent, str):
        sent = sent.split()
    
    vectors = []
    weights = []

    for token in sent:
        # Lấy vector embedding cho token
        vector = get_embed(token)
        if vector is not None:
            # Lấy tần suất của từ
            frequency = count_full.get(token, 10)
            weight = getLog(frequency)
            
            vectors.append(vector)
            weights.append(weight)
    
    if vectors:
        # Tính vector trung bình có trọng số
        doc_vector = np.average(vectors, weights=weights, axis=0)
    else:
        # Nếu không có vector hợp lệ, trả về vector 0
        doc_vector = np.zeros(model.vector_size)
    
    return doc_vector


def get_weighted_avg_data(article, get_vector_avg_weighted):
    data, keys = [], []
    for k, v in tqdm.tqdm(article.items()):
        # print v, len(v)
        # Anwen Hu 2019/8/5
        v = v['sentence']
        # print v, len(v)
        if len(v) < sen_len + 1:
            temp = np.zeros([300, len(v)])
            for i, sents in enumerate(v):
                # print sents
                temp[:, i] = get_vector_avg_weighted(sents.lower())
            # exit(0)
        else:
            temp = np.zeros([300, sen_len + 1])
            for i, sents in enumerate(v[:sen_len]):
                temp[:, i] = get_vector_avg_weighted(sents.lower())
            temp[:, sen_len] = np.average([get_vector_avg_weighted(sents.lower()) for sents in v[sen_len:]])
        data.append(temp)
        keys.append(k)

    return keys, data


if __name__ == '__main__':
    model = VnCoreNLP(save_dir='/content/', annotators=["wseg","ner"], max_heap_size='-Xmx4g')
    model_embed = fasttext.load_model('/content/cc.vi.300.bin')  # Load mô hình
    dataset = 'ViWiki'  # breakingewns/goodnews
    np.random.seed(42)
    # nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])
    print('nlp model loaded')
    count_full = Counter()
    if dataset == 'breakingnews':
        sen_len = 62
    else:
        sen_len = 54
    article_json_path = '/content/ICECAP/'+dataset+'_data/'+dataset+'_article.json'
    full = open_json(article_json_path)
    print('loaded', article_json_path)
    full_num = len(full.values())
    tokenized_num = 0
    for v_fu in full.values():
        for elm in v_fu['sentence']:
            tokens = list(set(t.lower() for t in model.word_segment(elm)[0].split(' ')))
            count_full.update(tokens)
        tokenized_num += 1
        sys.stdout.write(
            '\r Tokenizing articless: %d/%d articles processed...' % (tokenized_num, full_num))
        sys.stdout.flush()
    
    keys, data = get_weighted_avg_data(full, get_vector_avg_weighted_full)
    json.dump(keys, open('/content/ICECAP/'+dataset+'_data/'+dataset+'_articles_full_WeightedAvg_keys.json', 'w', encoding='utf-8'))
    save_h5('/content/ICECAP/'+dataset+'_data/'+dataset+'_articles_full_WeightedAvg.h5', data)
