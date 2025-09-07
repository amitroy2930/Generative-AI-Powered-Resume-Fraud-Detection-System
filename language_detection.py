import os
import sys
import pandas as pd
import re
# import fasttext

def parse_arguments():
    """
    Parse command-line arguments for the output path.
    """
    if len(sys.argv) < 3:
        print("Usage: python data_extraction.py <output_path>")
        sys.exit(1)
    return sys.argv[1], sys.argv[2]

def load_raw_data(file_path):
    """
    Load raw text data from the provided CSV file.
    """
    return pd.read_csv(file_path)

def clean_text(raw_text):
    """
    Clean a given text by removing URLs, emails, special characters, and newlines.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    text = url_pattern.sub('', raw_text)
    text = email_pattern.sub('', text)
    text = text.replace('\n', ' ')
    return re.sub(r'[^\w\s]', '', text)

# def detect_language(model, text):
#     """
#     Detect the language of a given text using a FastText model.
#     Returns the language and its confidence score.
#     """
#     prediction = model.predict(text, k=1)
#     language = prediction[0][0]
#     confidence = prediction[1][0]
#     return language, confidence

def process_data(df):
    """
    Process the raw text data: clean it and detect language and confidence.
    """
    resource_ids, raw_texts, clean_texts, langs = [], [], [], []

    for i, row in df.iterrows():
        c_text = clean_text(row['raw_text'])
        lang = row['languages']
        if not pd.notnull(c_text) or not c_text:
            continue
        
        resource_ids.append(row['resource_id'])
        raw_texts.append(row['raw_text'])
        clean_texts.append(c_text)
        # lang, conf = detect_language(model, c_text)
        langs.append(lang)
        # confs.append(conf)
        # print(f"Row {i}: Language - {lang}, Confidence - {conf:.2f}")
    
    new_df = pd.DataFrame()
    new_df['resource_id'] = resource_ids
    new_df['raw_text'] = raw_texts
    new_df['clean_text'] = clean_texts
    new_df['languages'] = langs
    # new_df['confidence'] = confs

    return new_df

# def save_language_specific_files(df, output_folder):
#     """
#     Save the processed data into separate CSV files for each language.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     for lang in df['language'].unique():
#         lang_df = df[df['language'] == lang]
#         file_path = os.path.join(output_folder, f"resume_details_{lang}.csv")
#         lang_df.to_csv(file_path, index=False)
#         print(f"Saved {file_path}")

def save_language_files(df, output_folder):
    """
    Save the processed data into a csv file.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "resume_details.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}")

def main():
    # Parse the command-line argument for output path
    input_path, output_folder = parse_arguments()

    # Load the raw data
    raw_text_df = load_raw_data(input_path)

    # Load the language detection model
    # model = fasttext.load_model('lid.176.ftz')

    # Process the data: clean text and detect language
    processed_data = process_data(raw_text_df)

    # Save processed data into language-specific files
    # save_language_specific_files(processed_data, output_folder=output_folder)
    save_language_files(processed_data, output_folder=output_folder)

if __name__ == "__main__":
    main()