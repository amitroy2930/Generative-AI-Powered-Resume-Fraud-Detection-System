## Resume Clustering and Similarity Computation

This report outlines a hybrid approach to clustering resumes based on similarity. The methodology combines semantic embeddings for initial broad-scope similarity detection with token-based methods for refined, accurate clustering.

---

### **1. Data Preprocessing**

The first step involved loading resume data, including `resource_id`, raw text, and language. The raw text was then cleaned by removing special characters, URLs, and common **stopwords** to prepare it for analysis.

---

### **2. Explored Approaches**

#### **2.1 Token-Based Similarity**

Traditional token-based methods like **Jaccard similarity** were deemed impractical for a large dataset of over 1.5 lakh resumes across 50+ languages. The primary challenges were the inability to capture **semantic similarity** and the massive vocabulary size (over 3 crore unique words), which made creating a dictionary computationally infeasible.

#### **2.2 Semantic Similarity**

To address the limitations of the token-based approach, an embedding-based method was adopted.

* **Numerical Representation of Text**: The cleaned text was converted into numerical **embeddings** using **Sentence Transformer (multilingual-MiniLM-L12-v2)**. This model was chosen for its multilingual capability and small size (∼400 MB).

* **Efficient Similarity Computation using FAISS**: Computing cosine similarity for every resume pair was too expensive. To overcome this, resume embeddings were stored using the **FAISS** library, and **K-Nearest Neighbors (KNN)** was applied to find the top K matching resumes, significantly improving efficiency.

* **Graph-Based Clustering**: For larger clusters, a **graph-based approach** was used. Resumes were treated as nodes, and connections were made based on similarity. This method linked and merged related groups, assigning them the same cluster ID.

#### **2.3 Addressing False Positives and Negatives**

Relying solely on semantic embeddings, KNN, and cosine similarity proved insufficient.

* A high similarity threshold (98%) led to **False Negatives** (missed similar resumes).
* A low threshold (90%) led to **False Positives** (unrelated resumes clustered together), resulting in impractical single clusters with up to 10,000 resumes. 

---

### **3. Hybrid Approach**

A **hybrid approach** was developed to leverage the strengths of both semantic and token-based methods. This two-stage process uses semantic embeddings for initial grouping and token-based methods for subsequent refinement.

#### **3.1 Cluster Refinement**

Initial matching was performed using a lower embedding similarity threshold (90-95%) with **KNN** to find the top K similar resumes. **DBSCAN** with **TF-IDF** was then applied to refine these clusters. While other methods were tested, DBSCAN often misclassified resumes as noise, leading to **False Negatives**. This refinement process effectively reduced the number of resumes in the largest cluster from ∼25,000 to ∼7,500.

#### **3.2 Enhancing Cluster Formation with TF-IDF Threshold**

A more effective refinement strategy was implemented by introducing a **TF-IDF threshold**. After identifying the top K matches using embeddings, a **TF-IDF similarity** check was performed on this smaller subset. This ensured that only resumes with both high semantic and high keyword-based similarity were clustered together. The **TF-IDF** method was chosen for its ability to capture word importance and because it was computationally feasible on small subsets. This approach replaced the inefficient DBSCAN refinement.

The final algorithm relies on three key hyperparameters:
1.  **K** in KNN
2.  **Embedding similarity threshold**
3.  **TF-IDF similarity threshold**

---

### **4. Final Parameter Optimization**

* **Optimal Value of K**: After testing, **K=5** was chosen. Increasing K to 100 introduced more irrelevant resumes and bloated the corpus, which reduced the effectiveness of TF-IDF. The average cluster size was determined to be 3 resumes, making a small K value sufficient.
* **Embedding Threshold**: The fixed embedding threshold was eliminated in favor of a KNN-based approach that retrieves the top K matches regardless of their similarity percentage. This was done to avoid missing resumes with lower similarity scores.
* **TF-IDF Threshold**: An optimal threshold of **0.6** was determined through experimentation, providing the best results.

---

### **5. Experimental Results**

The experiments were conducted on **142,059 resumes**. The final optimized parameters **(Embedding Threshold: 0.0, TF-IDF Threshold: 0.6, K: 5)** yielded the best balance of accuracy and efficiency.

| Embedding Threshold | TF-IDF Threshold | K | Total Similar Resumes | Percentage of Similar Resumes (%) | Total Clusters | False Positive | False Negative |
| ------------------- | ---------------- | - | ------------------- | --------------------------------- | -------------- | -------------- | -------------- |
| 0.9                 | 0.6              | 5 | 10,070              | 7.08                              | 3,203          | 0              | Many           |
| 0.0                 | 0.5              | 100 | 13,971              | 9.83                              | 3,944          | 1,508          | Very less      |
| 0.0                 | 0.6              | 100 | 12,186              | 8.58                              | 3,660          | 628            | Many           |
| **0.0** | **0.6** | **5** | **12,463** | **8.77** | **3,679** | **590** | **Very Minimal** |

The final result, with a low K and a high TF-IDF threshold, ensured that only the most relevant resumes were clustered, effectively minimizing both false positives and false negatives.

---

### **6. Conclusion**

The finalized **hybrid method** combines the semantic understanding of **embeddings** with the precision of **token-based refinement** to achieve accurate and efficient resume clustering. This approach successfully addresses the challenges of large, multilingual datasets by balancing computational efficiency with high clustering accuracy.