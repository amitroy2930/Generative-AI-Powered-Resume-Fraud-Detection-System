import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import time
import ast

nltk.download('stopwords')

# Language mappings for stopwords
# language_mapping = {
#     'ar': 'arabic',
#     'de': 'german',
#     'en': 'english',
#     'es': 'spanish',
#     'fr': 'french',
#     'id': 'indonesian',
#     'it': 'italian',
#     'ja': 'japanese',
#     'nl': 'dutch',
#     'pt': 'portuguese',
#     'sv': 'swedish',
#     'tr': 'turkish',
#     'zh': 'chinese'
# }

# supported_languages = list(language_mapping.keys())
# stop_words_dict = {
#     lang: stopwords.words(language_mapping[lang])
#     if language_mapping[lang] in stopwords.fileids()
#     else [] for lang in supported_languages
# }
japanese = ["あそこ", "あっ", "あの", "あのかた", "あの人", "あり", "あります", "ある", "あれ", "い", "いう", "います", "いる", "う", "うち", "え", "お", "および", "おり", "おります", "か", "かつて", "から", "が", "き", "ここ", "こちら", "こと", "この", "これ", "これら", "さ", "さらに", "し", "しかし", "する", "ず", "せ", "せる", "そこ", "そして", "その", "その他", "その後", "それ", "それぞれ", "それで", "た", "ただし", "たち", "ため", "たり", "だ", "だっ", "だれ", "つ", "て", "で", "でき", "できる", "です", "では", "でも", "と", "という", "といった", "とき", "ところ", "として", "とともに", "とも", "と共に", "どこ", "どの", "な", "ない", "なお", "なかっ", "ながら", "なく", "なっ", "など", "なに", "なら", "なり", "なる", "なん", "に", "において", "における", "について", "にて", "によって", "により", "による", "に対して", "に対する", "に関する", "の", "ので", "のみ", "は", "ば", "へ", "ほか", "ほとんど", "ほど", "ます", "また", "または", "まで", "も", "もの", "ものの", "や", "よう", "より", "ら", "られ", "られる", "れ", "れる", "を", "ん", "何", "及び", "彼", "彼女", "我々", "特に", "私", "私達", "貴方", "貴方方"]

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
    
def get_all_stop_words(languages):
    """
    Get the union of stop words for the given languages.
    """
    all_stop_words = set()
    for lang in languages:
        language = lang.lower()
        if language == 'japanese':
            all_stop_words.update(japanese)
        elif language in stopwords.fileids():
            all_stop_words.update(stopwords.words(language))
    
    return set(word.lower().strip() for word in all_stop_words)

def remove_stop_words(text, stop_words):
    """
    Remove stop words from a given text.
    """
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def calculate_tfidf_distance(gr_indices, json_data, resume_data):
    """
    Calculates cosine similarity between TF-IDF vectors for a group of indices.
    """
    try:
        clean_texts = []
        language_list = set()
        for index in gr_indices:
            text_row = resume_data[resume_data['resource_id'] == json_data[str(index)]]
            clean_text = text_row['clean_text'].values[0]
            languages = text_row['languages']
            language_list.update(ast.literal_eval(str(languages.values[0])))
            clean_texts.append(clean_text)
        
        if len(clean_texts) < len(gr_indices):  # If insufficient valid texts, return default
            return [0] * len(gr_indices)
        
        stop_words = get_all_stop_words(list(language_list))
        cleaned_texts = [remove_stop_words(text, stop_words) for text in clean_texts]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        tfidf_array = tfidf_matrix.toarray()
        reference_vector = tfidf_array[0]
        cosine_similarities = cosine_similarity(tfidf_array, reference_vector.reshape(1, -1))
        return cosine_similarities.flatten()
    except Exception as e:
        print(f"Error calculating TF-IDF distance: {e}")
        return [0] * len(gr_indices)


def main():
    """
    Main function to process embedding files and calculate similar groups.
    """
    folder_path, output_folder, resume_details_folder, threshold, tfidf_threshold, k = parse_arguments()
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            print(f"Processing {file}")
            # language = file.split('_')[-2]
            resume_details_path = f"{resume_details_folder}/resume_details.csv"
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
                start_time = time.time()
                tfidf_similarities = calculate_tfidf_distance(indices[i], json_data, resume_data)
                end_time = time.time()
                duration = end_time - start_time
                print(f'Time Taken: {duration}')
                for j in range(k):
                    if i == indices[i, j]:
                        continue
                    if distances[i, j] >= threshold and tfidf_similarities[j] >= tfidf_threshold:
                        G.add_edge(i, indices[i, j])

            # similar_groups = [list(component) for component in nx.connected_components(G) if len(component) > 1]
            similar_groups = [list(component) for component in nx.connected_components(G)]
            similar_groups_dict = {
                f"group_{i}": convert_index_to_user(group, json_data) for i, group in enumerate(similar_groups)
            }

            output_file = os.path.join(output_folder, "similar_groups.json")
            with open(output_file, 'w') as f:
                json.dump(similar_groups_dict, f)
            print(f"Finished processing {file}")


if __name__ == "__main__":
    main()