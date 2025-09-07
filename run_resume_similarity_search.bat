@echo off
set DATE=17_04_25
set "RESUME_DETAILS_PATH=.\data\extracted_data\resume_details_%DATE%.csv"
set "RESUME_DETAILS_OUTPUT_FOLDER=.\data\extracted_data\resume_details_%DATE%"
set "EMBEDDING_OUTPUT_FOLDER=.\data\resume_embeddings_%DATE%"
set "SIMILAR_GROUP_FOLDER=.\data\result\similar_groups_0_6_5_%DATE%"
set "FINAL_RES_PATH=.\data\final_res\final_result_0_6_5_%DATE%.csv"
set EMBEDDING_THRESHOLD=0.0
set TFIDF_THRESHOLD=0.6
set K=5

@REM Navigate to the directory containing the script
@REM cd /path/to/your/script/directory

:: Extract data from SQL Server
@REM python data_extraction.py "%RESUME_DETAILS_PATH%"


@REM :: Apply preprocessing and language detection
@REM python language_detection.py "%RESUME_DETAILS_PATH%" "%RESUME_DETAILS_OUTPUT_FOLDER%"

@REM ::Apple transformer-based embeddings
@REM python resume_embeddings.py "%RESUME_DETAILS_OUTPUT_FOLDER%" "%EMBEDDING_OUTPUT_FOLDER%"

::Make cluster of similar resume
@REM python resume_similarity_tf-idf.py "%EMBEDDING_OUTPUT_FOLDER%" "%SIMILAR_GROUP_FOLDER%" "%RESUME_DETAILS_OUTPUT_FOLDER%" "%EMBEDDING_THRESHOLD%" "%TFIDF_THRESHOLD%" "%K%"

@REM python resume_similarity_bm-25.py "%EMBEDDING_OUTPUT_FOLDER%" "%SIMILAR_GROUP_FOLDER%" "%RESUME_DETAILS_OUTPUT_FOLDER%" "%EMBEDDING_THRESHOLD%" "%TFIDF_THRESHOLD%" "%K%"

:: Final result will be saved in particular format
@REM python final_result_maker.py "%SIMILAR_GROUP_FOLDER%" "%RESUME_DETAILS_OUTPUT_FOLDER%" "%FINAL_RES_PATH%"

:: Final result with score will be saved in particular format
python final_result_with_score_maker.py "%SIMILAR_GROUP_FOLDER%" "%RESUME_DETAILS_OUTPUT_FOLDER%" "%FINAL_RES_PATH%" "%EMBEDDING_OUTPUT_FOLDER%"