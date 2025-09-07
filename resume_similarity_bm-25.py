import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
from rank_bm25 import BM25Okapi

nltk.download('stopwords')


# Language mappings for stopwords
language_mapping = {
    'ar': 'arabic',
    'de': 'german',
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'id': 'indonesian',
    'it': 'italian',
    'ja': 'japanese',
    'nl': 'dutch',
    'pt': 'portuguese',
    'sv': 'swedish',
    'tr': 'turkish',
    'zh': 'chinese'
}

supported_languages = list(language_mapping.keys())
stop_words_dict = {
    lang: stopwords.words(language_mapping[lang])
    if language_mapping[lang] in stopwords.fileids()
    else [] for lang in supported_languages
}
stop_words_dict['ja'] = ["あそこ", "あっ", "あの", "あのかた", "あの人", "あり", "あります", "ある", "あれ", "い", "いう", "います", "いる", "う", "うち", "え", "お", "および", "おり", "おります", "か", "かつて", "から", "が", "き", "ここ", "こちら", "こと", "この", "これ", "これら", "さ", "さらに", "し", "しかし", "する", "ず", "せ", "せる", "そこ", "そして", "その", "その他", "その後", "それ", "それぞれ", "それで", "た", "ただし", "たち", "ため", "たり", "だ", "だっ", "だれ", "つ", "て", "で", "でき", "できる", "です", "では", "でも", "と", "という", "といった", "とき", "ところ", "として", "とともに", "とも", "と共に", "どこ", "どの", "な", "ない", "なお", "なかっ", "ながら", "なく", "なっ", "など", "なに", "なら", "なり", "なる", "なん", "に", "において", "における", "について", "にて", "によって", "により", "による", "に対して", "に対する", "に関する", "の", "ので", "のみ", "は", "ば", "へ", "ほか", "ほとんど", "ほど", "ます", "また", "または", "まで", "も", "もの", "ものの", "や", "よう", "より", "ら", "られ", "られる", "れ", "れる", "を", "ん", "何", "及び", "彼", "彼女", "我々", "特に", "私", "私達", "貴方", "貴方方"]


def parse_arguments():
    """
    Parse and validate command-line arguments.
    Returns:
        tuple: folder_path, output_folder, threshold, tfidf_threshold
    """
    if len(sys.argv) != 7:
        print("Usage: python script.py <folder_path> <output_folder> <threshold> <tfidf_threshold>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_folder = sys.argv[2]
    resume_details_folder = sys.argv[3]
    threshold = float(sys.argv[4])
    tfidf_threshold = float(sys.argv[5])
    k = int(sys.argv[6])
    
    return folder_path, output_folder, resume_details_folder, threshold, tfidf_threshold, k


def convert_index_to_user(obj, json_data):
    """
    Converts indices to user-friendly format using JSON data.
    """
    if isinstance(obj, (np.int64, int)):
        return int(obj), int(json_data.get(str(obj), obj))
    elif isinstance(obj, list):
        return [convert_index_to_user(item, json_data) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_index_to_user(value, json_data) for key, value in obj.items()}
    else:
        return obj


def calculate_tfidf_distance(gr_indices, json_data, resume_data, language):
    """
    Calculates cosine similarity between TF-IDF vectors for a group of indices.
    """
    clean_texts = []
    for index in gr_indices:
        text_row = resume_data[resume_data['resource_id'] == json_data[str(index)]]
        if not text_row.empty:
            clean_text = text_row['clean_text'].values[0]
            if pd.notnull(clean_text):
                clean_texts.append(clean_text)
    
    if len(clean_texts) < len(gr_indices):  # If insufficient valid texts, return default
        return [0] * len(gr_indices)
    
    stop_words = stop_words_dict.get(language, stop_words_dict['en'])
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    tfidf_array = tfidf_matrix.toarray()
    reference_vector = tfidf_array[0]
    cosine_similarities = cosine_similarity(tfidf_array, reference_vector.reshape(1, -1))
    return cosine_similarities.flatten()

def calculate_bm25_distance(gr_indices, json_data, resume_data, language):
    """
    Calculates BM25 similarities for a group of indices.
    """
    clean_texts = []
    for index in gr_indices:
        text_row = resume_data[resume_data['resource_id'] == json_data[str(index)]]
        if not text_row.empty:
            clean_text = text_row['clean_text'].values[0]
            if pd.notnull(clean_text):
                clean_texts.append(clean_text)
    
    if len(clean_texts) < len(gr_indices):  # If insufficient valid texts, return default
        return [0] * len(gr_indices)
    
    # Tokenize texts for BM25
    stop_words = stop_words_dict.get(language, stop_words_dict['en'])
    tokenized_texts = [
        [word for word in text.split() if word not in stop_words] for text in clean_texts
    ]

    # BM25 implementation
    bm25 = BM25Okapi(tokenized_texts)
    reference_vector = tokenized_texts[0]
    bm25_scores = bm25.get_scores(reference_vector)
    return bm25_scores

def calculate_bm25_distance_cosine(gr_indices, json_data, resume_data, language):
    """
    Calculates cosine similarity between BM25 vectors for a group of indices.
    """
    clean_texts = []
    for index in gr_indices:
        text_row = resume_data[resume_data['resource_id'] == json_data[str(index)]]
        if not text_row.empty:
            clean_text = text_row['clean_text'].values[0]
            if pd.notnull(clean_text):
                clean_texts.append(clean_text)
    
    if len(clean_texts) < len(gr_indices):  # If insufficient valid texts, return default
        return [0] * len(gr_indices)
    
    # Tokenize texts for BM25
    stop_words = stop_words_dict.get(language, stop_words_dict['en'])
    tokenized_texts = [
        [word for word in text.split() if word not in stop_words] for text in clean_texts
    ]
    
    # BM25 implementation
    bm25 = BM25Okapi(tokenized_texts)
    bm25_vectors = [
        [bm25.idf.get(word, 0) for word in bm25.idf.keys()] for _ in tokenized_texts
    ]


    # Build the document-term matrix for BM25
    bm25_matrix = np.array(bm25_vectors)
    reference_vector = bm25_matrix[0]

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(bm25_matrix, reference_vector.reshape(1, -1))
    return cosine_similarities.flatten()

def main():
    """
    Main function to process embedding files and calculate similar groups.
    """
    folder_path, output_folder, resume_details_folder, threshold, tfidf_threshold, k = parse_arguments()
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            print(f"Processing {file}")
            language = file.split('_')[-2]
            resume_details_path = f"{resume_details_folder}/resume_details___label__{language}.csv"
            resume_data = pd.read_csv(resume_details_path)

            json_file = os.path.join(folder_path, file.replace('_embeddings.npy', '_index_to_user_map.json'))
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            embeddings = np.load(os.path.join(folder_path, file))
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

            # k = 5
            distances, indices = index.search(embeddings, k)

            G = nx.Graph()
            for i in range(indices.shape[0]):
                bm25_similarities = calculate_bm25_distance_cosine(indices[i], json_data, resume_data, language)
                for j in range(1, k):
                    if distances[i, j] >= threshold and bm25_similarities[j] >= tfidf_threshold:
                        G.add_edge(i, indices[i, j])

            similar_groups = [list(component) for component in nx.connected_components(G) if len(component) > 1]
            similar_groups_dict = {
                f"group_{i}": convert_index_to_user(group, json_data) for i, group in enumerate(similar_groups)
            }

            output_file = os.path.join(output_folder, f"{language}_similar_groups.json")
            with open(output_file, 'w') as f:
                json.dump(similar_groups_dict, f)
            print(f"Finished processing {file}")


if __name__ == "__main__":
    main()