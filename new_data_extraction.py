import csv
import pandas as pd
from utils.connect_to_db import db_connect
import json
import sys
import gc
import os


def parse_arguments():
    """
    Parse command-line arguments for output path.
    """
    if len(sys.argv) < 2:
        print("Usage: python data_extraction.py <output_path>")
        sys.exit(1)
    return sys.argv[1]

def save_json_to_files(batch_df, output_folder):
    """
    Save each JSON string from the DataFrame into separate files.
    Files are named as 1.json, 2.json, etc.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the folder if it doesn't exist

    for idx, row in batch_df.iterrows():
        file_path = os.path.join(output_folder, f"{row['resource_id']}.json")  # File name starts from 1.json
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(row['curriculum_json']) 

def get_work_exs_and_educations(resume_data):
    work_exs = ""
    educations = ""
    if 'sections' not in resume_data:
        return "", ""
    for item in resume_data['sections']:
        if item['sectionType'] == "WorkExperience":
            work_exs += item['text']
        if item['sectionType'] == "Education":
            educations += item['text']

    return work_exs, educations

def get_raw_text_and_resume_probability(text):
    """
    Extract raw text and resume probability from a JSON string.
    """
    try:
        data = json.loads(text)
        raw_text = data.get('rawText', '').strip()
        languages = data.get('languages', '')
        # resume_probability = int(data.get('isResumeProbability', '-1'))
        # work_exs, educations = get_work_exs_and_educations(data)
        return raw_text, languages
    except:
        return "raw text not present", []
    

def get_raw_text(text):
    """
    Extract raw text and resume probability from a JSON string.
    """
    try:
        return text.strip()
    except:
        return "raw text not present"

def process_batch(batch_df):
    """
    Process a single batch of data to extract raw text.
    """
    batch_df[['raw_text']] = batch_df['curriculum_data'].apply(
        lambda x: pd.Series(get_raw_text(x))
    )

    no_raw_text_parse_df = batch_df[batch_df['raw_text'] == "raw text not present"]
    print(f"Percentage of data for which parsable raw text is not present: {(len(no_raw_text_parse_df)/len(batch_df)) * 100:.2f}%")

    no_raw_text_df = batch_df[batch_df['raw_text'] == ""]
    print(f"Percentage of data for which raw text is not present: {(len(no_raw_text_df)/len(batch_df)) * 100:.2f}%")

    return batch_df[~batch_df['raw_text'].isin(['', "raw text not present"])]


def save_to_csv(df, writer):
    """
    Save processed data to the CSV file.
    """
    # df[['resource_id', 'raw_text', 'languages']].to_csv(writer, index=False, header=False)
    # df[['resource_id', 'raw_text']].to_csv(writer, index=False, header=False)
    df[['resource_id', 'raw_text']].to_csv(writer, index=False, header=False, escapechar='\\')



def main():
    """
    Main function to extract and save resume data.
    """
    # output_path = parse_arguments()
    output_path = "./data/extracted_data/resume_details_cd_15_04_25.csv"
    resume_engine = db_connect("corpus2_db")
    batch_size = 1000
    offset = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writerow(['resource_id', 'raw_text', 'languages'])
        writer.writerow(['resource_id', 'raw_text'])

        while True:
            # query = f"""
            #     SELECT resource_id, curriculum_json
            #     FROM curriculum_data
            #     WHERE curriculum_json IS NOT NULL
            #     LIMIT {batch_size} OFFSET {offset}
            # """
            query = f"""
                SELECT resource_id, curriculum_data
                FROM curriculum_data
                WHERE curriculum_data IS NOT NULL AND curriculum_data != ''
                LIMIT {batch_size} OFFSET {offset}
            """
            batch_df = pd.read_sql(query, con=resume_engine)

            if batch_df.empty:
                break
            
            # output_folder = './data/json_files/'
            # save_json_to_files(batch_df, output_folder)
            
            processed_df = process_batch(batch_df)
            save_to_csv(processed_df, csv_file)
            
            # del batch_df, processed_df
            del processed_df
            gc.collect()

            offset += batch_size

    print("Processing complete.")


if __name__ == "__main__":
    main()