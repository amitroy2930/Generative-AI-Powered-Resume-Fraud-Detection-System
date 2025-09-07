import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
import faiss
import sys

def load_embeddings(embeddings_path):
    path = os.path.join(embeddings_path, "resume_details_embeddings.npy")
    embeddings = np.load(path)
    faiss.normalize_L2(embeddings)
    return embeddings

def compute_cluster_metrics(cluster_indices, embeddings, threshold=0.9):
    cluster_embeddings = embeddings[cluster_indices]
    similarity_matrix = cosine_similarity(cluster_embeddings)
    k = len(cluster_indices)
    similarities = [similarity_matrix[i, j] for i in range(k) for j in range(i + 1, k)]

    avg_similarity = mean(similarities) if similarities else 0
    min_similarity = min(similarities) if similarities else 0
    max_similarity = max(similarities) if similarities else 0
    below_threshold_count = sum(1 for sim in similarities if sim < threshold)
    percentage_above = 1 - (below_threshold_count / len(similarities)) if similarities else 0

    return {
        "size": k,
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "below_threshold_count": below_threshold_count,
        "percentage_of_above_threshold_count": percentage_above
    }

def compute_all_cluster_scores(similar_group_path, embeddings):
    data_before = {}

    for file_name in os.listdir(similar_group_path):
        language = file_name.split('_')[0]
        file_path = os.path.join(similar_group_path, file_name)

        with open(file_path, 'r') as f:
            cluster_data = json.load(f)

        if language not in data_before:
            data_before[language] = {}

        for group_id, group_members in cluster_data.items():
            cluster_indices = [item[0] for item in group_members]
            metrics = compute_cluster_metrics(cluster_indices, embeddings)
            score = (
                0.3 * metrics['avg_similarity'] +
                0.3 * metrics['percentage_of_above_threshold_count'] +
                0.2 * metrics['max_similarity'] +
                0.2 * metrics['max_similarity']
            )
            data_before[language][group_id] = score

    return data_before

def load_resume_data(resume_details_folder):
    file_path = os.path.join(resume_details_folder, "resume_details.csv")
    if not os.path.exists(file_path):
        print(f"Resume details file not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def enrich_data_with_resume(similar_group_path, data_before, resume_data):
    final_res = {}

    for file_name in os.listdir(similar_group_path):
        language = file_name.split('_')[0]
        file_path = os.path.join(similar_group_path, file_name)

        with open(file_path, 'r') as f:
            cluster_data = json.load(f)

        for group_id, group_members in cluster_data.items():
            cluster_id = int(group_id.split('_')[-1])
            for member in group_members:
                resource_id = member[1]
                row = resume_data[resume_data['resource_id'] == resource_id]

                if row.empty:
                    print(f"Resource ID {resource_id} not found.")
                    continue

                raw_text = row['raw_text'].iloc[0]
                languages = row['languages'].iloc[0]
                if language not in final_res:
                    final_res[language] = []

                final_res[language].append({
                    "resource_id": resource_id,
                    "raw_text": raw_text,
                    "cluster_id": cluster_id,
                    "languages": languages,
                    "score": data_before[language][group_id]
                })

    return final_res

def save_results(final_res, pkl_path, csv_path):
    df = pd.concat([pd.DataFrame(recs) for recs in final_res.values()])
    df_sorted = df.sort_values(by=['score', 'cluster_id'], ascending=[True, True])
    df_sorted.to_pickle(pkl_path)
    df_sorted.to_csv(csv_path, index=False)
    print("Saved results to CSV and PKL.")

def main():
    # Config
    if len(sys.argv) != 5:
        print("Usage: python script.py <similar_group_path> <resume_details_folder> <final_res_path> <embeddings_path>")
        sys.exit(1)

    # Assign paths from command line arguments
    similar_group_path = sys.argv[1]
    resume_details_folder = sys.argv[2]
    final_res_path_csv = sys.argv[3]
    final_res_path_pkl = final_res_path_csv.replace('.csv', '.pkl')
    embeddings_path = sys.argv[4]

    print("Loading resume embeddings...")
    embeddings = load_embeddings(embeddings_path)

    print("Computing cluster scores...")
    data_before = compute_all_cluster_scores(similar_group_path, embeddings)

    print("Loading resume data...")
    resume_data = load_resume_data(resume_details_folder)
    if resume_data is None:
        return

    print("Enriching clusters with resume info...")
    final_res = enrich_data_with_resume(similar_group_path, data_before, resume_data)

    print("Saving results...")
    save_results(final_res, final_res_path_pkl, final_res_path_csv)

if __name__ == "__main__":
    main()