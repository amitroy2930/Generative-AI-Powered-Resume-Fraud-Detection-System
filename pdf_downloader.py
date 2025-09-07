import pandas as pd
from utils.connect_to_db import db_connect
import os
from azure.storage.blob import BlobServiceClient
from utils.common import get_hash_value

resume_engine = db_connect("corpus2_db")

def get_cv_url(resource_id):
    query = f"""select resource_id, country_id, cv_url 
            from corpus2_db.users_active_info 
            where resource_id = "{resource_id}"
            """

    resume_df = pd.read_sql(query, con=resume_engine)
    country_id = resume_df['country_id'].iloc[0]
    cv_url = resume_df['cv_url'].iloc[0]

    return cv_url

def connect_to_azure_storage_account(storage_account_name, storage_account_key):
    """
    Function to connect ot azure blob storage
    :param storage_account_name: storage account name
    :param storage_account_key: secret key to make connection to the storage
    :return: blob service client object
    """

    # Creating account url
    account_url = "https://{}.blob.core.windows.net".format(storage_account_name)

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=storage_account_key)

    return blob_service_client


def download_blob_to_file(blob_service_client, container_name, blob_path, local_blob_save_path, resource_id):
    """
    Function to download a blob file to local
    :param blob_service_client: BlobServiceClient object
    :param container_name: name of the container
    :param blob_path: path of the file on blob
    :param local_blob_save_path: Local path where the file needs to be downloaded
    :return:
    """

    # Get the blob object
    blob_client = blob_service_client.get_blob_client(container=container_name,
                                                      blob=blob_path)

    # Check if local save path exists
    os.makedirs(local_blob_save_path, exist_ok=True)

    # full file path of local
    # local_file_path = os.path.join(local_blob_save_path, blob_path)
    local_file_path = os.path.join(local_blob_save_path, f'{resource_id}.docx')

    # Download file to local path
    with open(file=local_file_path, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())

if __name__ == "__main__":

    # Load env variables
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    resource_id = 5318161

    storage_account_name = os.environ.get("AZURE_BLOB_ACCOUNT_NAME")
    storage_account_key = os.environ.get("AZURE_BLOB_ACCOUNT_KEY")

    # required parameters for functon call
    container_name = get_hash_value(f"{resource_id}")
    blob_path = get_cv_url(resource_id)
    local_blob_save_path = "./data/pdf_files"

    # Connect to Blob Service
    blob_service_client = connect_to_azure_storage_account(storage_account_name, storage_account_key)

    # Function call
    download_blob_to_file(blob_service_client, container_name, blob_path, local_blob_save_path, resource_id)