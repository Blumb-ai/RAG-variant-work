
def execute_drive_script():
    import io
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    from dotenv import load_dotenv
    import os
    import json

    load_dotenv()  # Carga las variables de entorno del archivo .env

    SCOPES = [os.getenv('SCOPES')]
    SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')


    credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('drive', 'v3', credentials=credentials)


    folder_id = os.getenv('FOLDER_DRIVE_ID') #Carpeta de drive
    query = f"'{folder_id}' in parents"

    response = service.files().list(q=query, fields="files(id, name, mimeType, webViewLink)").execute()
    files = response.get('files', [])
    source_documents = {}

    directory = "./documentos/"
    os.makedirs(directory, exist_ok=True)


    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.document':
            request = service.files().export_media(fileId=file['id'], mimeType='application/pdf')
            filename = f"{file['name']}.pdf"  # Guardar como PDF
            # Añade la URL del documento a source_documents
            source_documents[filename] = file.get('webViewLink', 'URL no disponible')

        elif file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            request = service.files().export_media(fileId=file['id'], mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            filename = f"{file['name']}.xlsx"  # Guardar como Excel
            source_documents[filename] = file.get('webViewLink', 'URL no disponible')
        elif file['mimeType'] == 'application/vnd.google-apps.presentation':
            request = service.files().export_media(fileId=file['id'], mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation')
            filename = f"{file['name']}.pptx"  # Guardar como PowerPoint
            source_documents[filename] = file.get('webViewLink', 'URL no disponible')
        else:
            request = service.files().get_media(fileId=file['id'])
            filename = file['name']
            source_documents[filename] = file.get('webViewLink', 'URL no disponible')

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        filename = directory + filename
        # El contenido del archivo está en fh.getvalue(). Guarda este contenido en un archivo local.
        with open(filename, 'wb') as f:
            f.write(fh.getvalue())
            print(f"Archivo '{filename}' descargado.")
        # El contenido del archivo está en fh.getvalue(). Guarda este contenido en un archivo local.
            
    with open('./documentos/source_documents.json', 'w') as fp:
        json.dump(source_documents, fp)

    print("Source documents guardados.")
    return source_documents