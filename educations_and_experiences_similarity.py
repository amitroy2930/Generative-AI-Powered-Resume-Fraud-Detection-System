import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import time
import re

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

def get_clean_text(raw_text):
    """
    Clean a given text by removing URLs, emails, special characters, and newlines.
    Includes error handling with try and except.
    """
    try:
        # Compile regex patterns
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Remove URLs
        text = url_pattern.sub('', raw_text)
        # Remove emails
        text = email_pattern.sub('', text)
        # Replace newlines with space
        text = text.replace('\n', ' ')
        # Remove special characters
        return re.sub(r'[^\w\s]', '', text)
    
    except Exception as e:
        # print(f"An error occurred: {e}")
        return ""

def calculate_tfidf_distance(gr_indices, resume_data, language, work_or_education = 'work_exs'):
    """
    Calculates cosine similarity between TF-IDF vectors for a group of indices.
    """
    clean_texts = []
    for index in gr_indices:
        text_row = resume_data[resume_data['resource_id'] == str(index)]
        
        if len(text_row[work_or_education].values) == 1:
            raw_text = text_row[work_or_education].values[0]
            clean_text = get_clean_text(raw_text)
            clean_texts.append([clean_text, index])
    
    stop_words = stop_words_dict.get(language, stop_words_dict['en'])
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    
    new_clean_texts = []
    for text_index in clean_texts:
        if text_index[0]:
            new_clean_texts.append(text_index)

    if len(new_clean_texts) < 2:
        return 0
    
    new_clean_texts = np.array(new_clean_texts)
    tfidf_matrix = vectorizer.fit_transform(new_clean_texts[:, 0])
    tfidf_array = tfidf_matrix.toarray()
    cosine_similarities = {}

    for i in range(len(tfidf_array)):
        reference_vector = tfidf_array[i]
        cosine_similarity_val = np.round(np.mean(cosine_similarity(tfidf_array, reference_vector.reshape(1, -1))), 4)
        cosine_similarities[str(new_clean_texts[i][1])] = cosine_similarity_val

    return cosine_similarities

def main():
    resume_groups_folder = './data/result/similar_groups_9_6'
    resume_details_path = './data/extracted_data/resume_details_new_test_1.csv'
    output_folder_path = './data/new_final_res'
    os.makedirs(output_folder_path, exist_ok=True)

    resume_data = pd.read_csv(resume_details_path, encoding='utf-8',low_memory=False)
    result_resume_data = pd.read_csv(f'./data/final_res/final_res_9_6.csv', usecols=[0, 1, 2, 3])
    result_resume_data['resource_id'] = result_resume_data['resource_id'].astype('str')

    work_exs_tfidf_score_dict = {}
    educations_tfidf_score_dict = {}


    for file in os.listdir(resume_groups_folder):
        language = file.split('_')[0]
        json_path = os.path.join(resume_groups_folder, file)

        with open(json_path, 'r') as f:
            file_content = json.load(f)
            for indices in file_content.values():
                gr_indices = [index[1] for index in indices]
                work_exs_tfidf_score = calculate_tfidf_distance(gr_indices, resume_data, language, 'work_exs')
                educations_tfidf_score = calculate_tfidf_distance(gr_indices, resume_data, language, 'educations')
                if work_exs_tfidf_score:
                    for index, score in work_exs_tfidf_score.items():
                        work_exs_tfidf_score_dict[index]=score
                if educations_tfidf_score:
                    for index, score in educations_tfidf_score.items():
                        educations_tfidf_score_dict[index]=score

    result_resume_data['work_exs_similarity_score'] = result_resume_data['resource_id'].map(work_exs_tfidf_score_dict)
    result_resume_data['educations_similarity_score'] = result_resume_data['resource_id'].map(educations_tfidf_score_dict)
    result_resume_data.to_csv(os.path.join(output_folder_path, f'final_res_9_6(k=5)_1.csv'), index=False)

if __name__ == "__main__":
    main()