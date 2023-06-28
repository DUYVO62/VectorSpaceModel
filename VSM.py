import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def load_data(path):
    with open(path, 'r') as f:
        data = f.read()
    ID_data = data.lower().replace("\n", " ").split("/")
    ID_data.pop()
    data = []
    for dat in ID_data:
        dat = dat.strip()
        id = dat.find(" ")
        data.append(dat[id+1:])
    return data

def remove_stopwords(docs):
    stop_words = set(stopwords.words('english'))
    filtered_docs = []
    for doc in docs:
        tokens = word_tokenize(doc)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        filtered_docs.append(' '.join(filtered_tokens))
    sentences = [s.split(' ') for s in filtered_docs]
    return sentences


def create_dict_words(docs):
    dict_words = []
    for doc in docs:
        for word in doc:
            if word not in dict_words:
                dict_words.append(word)
    return dict_words


def tf_words(docs, dict_words):
    matrix_tf = []
    for doc in docs:
        array_tf = []
        for word in dict_words:
            w_tf_value = 0
            tf = doc.count(word)

            if( tf > 0):
                w_tf_value = 1 + math.log10(tf)
            array_tf.append(w_tf_value)
        matrix_tf.append(array_tf)
    return matrix_tf


def idf_words(docs, dict_words):
    array_idf = []
    N = len(docs)
    for word in dict_words:
        df = sum([1 for doc in docs if word in doc])
        idf_value = math.log10(N/df)
        array_idf.append(idf_value)
    return array_idf


def tf_idf_words(matrix_tf, array_idf):
    matrix_tf_idf = []
    for array_tf in matrix_tf:
        array_tf_idf = []
        for i, tf_value in enumerate(array_tf):
            idf_value = array_idf[i]
            tf_idf_value = tf_value * idf_value
            array_tf_idf.append(tf_idf_value)
        matrix_tf_idf.append(array_tf_idf)
    return matrix_tf_idf


def tf_idf_query(query, array_idf, dict_words):
    query_array = [0] * len(dict_words)
    for i, word in enumerate(dict_words):
        if word in query:
            tf = query.count(word)
            tf_value = 1 + math.log10(tf)
            query_array[i] = tf_value * array_idf[i]
    return query_array 


def similarity_cosine(tf_idf_docs, tf_idf_queries):
    cosine = []
    for i, values in enumerate(tf_idf_docs):
        try:
            similarity_score = np.dot(values, tf_idf_queries) / (np.linalg.norm(values) * np.linalg.norm(tf_idf_queries))
        except RuntimeWarning:
            similarity_score = 0.0
        cosine.append([similarity_score, i])
        cosine.sort(reverse=True)
        top5_cosine = cosine[:5]
    return top5_cosine


def main():

    num = int(input("Nhập STT Câu Query (0 - 92): "))

    documents = load_data("./doc-text.txt")
    list_words_docs = remove_stopwords(documents)
    dict_docs = create_dict_words(list_words_docs)

    queries = load_data("./query-text.txt")
    list_words_queries = remove_stopwords(queries)
    query = list_words_queries[num]

    print('-------------------*.*.*.*.*-------------------')
    print(f'Query : {queries[num]}')
    print('-------------------*.*.*.*.*-------------------')


    tf_docs = tf_words(list_words_docs, dict_docs)
    idf_docs = idf_words(list_words_docs, dict_docs)
    tf_idf_docs = tf_idf_words(tf_docs, idf_docs)


    tf_idf_que = tf_idf_query(query, idf_docs, dict_docs)


    result = similarity_cosine(tf_idf_docs, tf_idf_que)
    for score, i in result:
        print(f'Cosine : {score: .4f}, Doc {i + 1}')

if __name__ == "__main__":
    main()
