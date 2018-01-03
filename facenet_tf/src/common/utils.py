import os, requests, zipfile

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(URL, destination):
    session = requests.Session()

    response = session.get(URL, params={'id': 1}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': 1, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_pre_model_path(pre_model_url, pre_model_zip, pre_model_path, pre_model_name):
    if not os.path.exists(pre_model_path):
        os.makedirs(pre_model_path)

    if os.path.isfile(pre_model_zip) == False:
        try:
            download_file_from_google_drive(pre_model_url, pre_model_zip)
        except:
            print("Error : facenet model down.")

    if os.path.isfile(pre_model_name) == False:
        fantasy_zip = zipfile.ZipFile(pre_model_zip)
        fantasy_zip.extractall(pre_model_path)
        fantasy_zip.close()