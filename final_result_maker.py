import json
import pandas as pd
import os
import sys

def load_resume_data(resume_details_folder, language):
    """
    Load the resume details CSV file for the specified language.
    """
    file_path = os.path.join(resume_details_folder, "resume_details.csv")
    if not os.path.exists(file_path):
        print(f"Resume details file not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def process_json_file(json_path, resume_data, group_idx, final_res):
    """
    Process a JSON file to extract and append resume details.
    """
    with open(json_path, 'r') as f:
        file_content = json.load(f)
    
    for _, groups in file_content.items():
        group_idx += 1
        for g in groups:
            resource_id = g[1]
            temp_data = resume_data[resume_data['resource_id'] == resource_id]
            if temp_data.empty:
                print(f"Resource ID {resource_id} not found in resume data.")
                continue
            temp_raw_text = temp_data['raw_text'].iloc[0]
            language = temp_data['languages'].iloc[0]
            final_res.append({
                "resource_id": resource_id,
                "raw_text": temp_raw_text,
                "cluster_id": group_idx,
                "language": language
            })
    return group_idx

def main():
    # Check if the script is called with the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <similar_group_path> <resume_details_folder> <final_res_path>")
        sys.exit(1)

    # Assign paths from command line arguments
    similar_group_path = sys.argv[1]
    resume_details_folder = sys.argv[2]
    final_res_path = sys.argv[3]
    final_res_path_pkl = final_res_path.replace('.xlsx', '.pkl')

    # Initialize variables
    final_res = []
    group_idx = -1

    # Process each file in the similar_group_path directory
    for file_name in os.listdir(similar_group_path):
        json_path = os.path.join(similar_group_path, file_name)
        language = file_name.split('_')[0]  # Extract language from file name

        resume_data = load_resume_data(resume_details_folder, language)
        if resume_data is None:
            continue  # Skip if resume data file is missing
        
        group_idx = process_json_file(json_path, resume_data, group_idx, final_res)

    # Convert the result list into a DataFrame
    df_final_res = pd.DataFrame(final_res)

    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(final_res_path), exist_ok=True)

    # Save the DataFrame to a CSV file
    # df_final_res.to_csv(final_res_path, index=False, encoding='utf-8-sig')
    # df_final_res.to_excel(final_res_path, index=False)
    df_final_res.to_pickle(final_res_path_pkl)

    # save df_final_res as pickle file


    print(f"Final result saved to {final_res_path}")

if __name__ == "__main__":
    main()