
def process_documents():
    import os
    import glob
    import fitz  # PyMuPDF para leer PDFs
    import pandas as pd
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from dotenv import load_dotenv
    import json

    # Carga las variables de entorno
    load_dotenv()  

    # Configuración del cliente de OpenAI
    if 'OPENAI_API_KEY' in os.environ:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    else:
        print("No se encontró la clave API de OpenAI. Asegúrate de haber establecido la variable de entorno 'OPENAI_API_KEY'.")
        exit(1)

    # Cargar source_documents del archivo JSON
    with open('./documentos/source_documents.json', 'r') as fp:
        source_documents = json.load(fp)

    # Configuración de directorios y archivos
    gfiles = glob.glob("./documentos/*.pdf")  # Procesar solo PDFs
    embeddings_dir = "./embeddings/"
    os.makedirs(embeddings_dir, exist_ok=True)  # Crea el directorio si no existe
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)

    # DataFrame para almacenar los embeddings y metadatos
    df_embeddings = pd.DataFrame(columns=['text', 'embedding', 'source'])

    for gfile in gfiles:
        try:
            # Abre el PDF y extrae el texto
            doc = fitz.open(gfile)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if not text:
                print(f"No se pudo extraer texto del archivo: {gfile}")
                continue        
            # Divide el texto en fragmentos
            texts = text_splitter.split_text(text)
            
            # Obtiene el nombre del archivo sin la ruta para buscar su URL
            filename = os.path.basename(gfile)
            document_source = source_documents.get(filename, 'URL no disponible')
            for text_fragment in texts:    
                        
                response = client.embeddings.create(input=text_fragment, model="text-embedding-ada-002")
                embedding = response.data[0].embedding
                new_row = pd.DataFrame({
                    'text': [text_fragment],
                    'embedding': [embedding],
                    'source': [document_source]
                })
            df_embeddings = pd.concat([df_embeddings, new_row], ignore_index=True)

            
            print(f"Documentos de '{filename}' procesados.")
        except Exception as e:
            print(f"Ocurrió un error al procesar el archivo {gfile}: {e}")

    print("Proceso completado.")

    # Guarda el DataFrame en un archivo CSV
    df_embeddings.to_csv(os.path.join(embeddings_dir, 'embeddings.csv'), index=False)
    df_ok = "Proceso completado."
    return df_ok
