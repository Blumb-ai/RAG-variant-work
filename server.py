from fastapi import FastAPI, Request, HTTPException
from langchain_openai import ChatOpenAI
import os
import pandas
import openai
import numpy as np
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts.prompt import PromptTemplate
from loadDocuments import execute_drive_script
from embeddingData import process_documents
from langchain_core.prompts.prompt import PromptTemplate
import fitz
from classifyDocuments import classify_documents
import openai
from pydantic import BaseModel
from typing import List


class Question(BaseModel):
    question: str

class ResponseModel(BaseModel):
    response: str
    source: str

class ErrorResponseModel(BaseModel):
    detail: str

class Document(BaseModel):
    content: str



load_dotenv()  

if 'OPENAI_API_KEY' in os.environ:
    client = openai.OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)
else:
    print("No se encontró la clave API de OpenAI. Asegúrate de haber establecido la variable de entorno 'OPENAI_API_KEY'.")
    exit(1)  # Sale del script si no se encuentra la clave API

ips = []
ips_times = []

ips_ref = []
ips_times_ref = []

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def split_into_segments(text, max_length):
    # Divide el texto en palabras
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length and current_segment:
            # Cuando el segmento actual alcanza la longitud máxima, lo guardamos y comenzamos uno nuevo
            segments.append(' '.join(current_segment))
            current_segment = [word]
            current_length = len(word)
        else:
            # Añade la palabra al segmento actual
            current_segment.append(word)
            current_length += len(word) + 1  # +1 para el espacio

    # Añade el último segmento si hay alguno
    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_title_or_first_chars(text, char_limit=200):
    # Extrae solo el título o los primeros caracteres del texto
    return text.strip().split('\n')[0][:char_limit]

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def logic(question):
    
    if os.path.exists("./embeddings/embeddings.csv"):
        df = pandas.read_csv("./embeddings/embeddings.csv")
    else:
        print("Archivo CSV no encontrado")

    embs = []
    for r1 in range(len(df.embedding)): # Changing the format of the embeddings into a list due to a parsing error
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding( 
        question
    ) # Creating an embedding for the question that's been asked

    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding)) # Finds the relevance of each piece of data in context of the question
    df.to_csv("./embeddings/embeddings.csv")

    df2 = df.sort_values("similarity", ascending=False) # Sorts the text chunks based on how relevant they are to finding the answer to the question
    df2.to_csv("./embeddings/embeddings.csv")
    df2 = pandas.read_csv("./embeddings/embeddings.csv")
    if df2.empty:
        print("df2 está vacío")
    else:
        df2["similarity"][0]

    from langchain.docstore.document import Document

    # Obtener los datos relevantes
    if not df2.empty:
        comb = [df2["text"][0]]
        source = df2["source"].iloc[0]  # Guardar la fuente del embedding más relevante
        docs = [Document(page_content=t) for t in comb]  # Obtener el fragmento de texto más relevante
    else:
        return "No se encontraron datos relevantes para la respuesta."
    
    prompt_template = question + "\n\n{text}\n\n" 
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)  # Preparar el LLM

    output = chain.invoke(docs)  # Formular una respuesta
    print('ESTO ES OUTPUT')
    print(output)

    if isinstance(output, dict) and 'output_text' in output:
        response_text = output['output_text']
    else:
        response_text = "La respuesta no está en el formato esperado."
    
    return response_text, source

app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})

@app.post('/ask', response_model=ResponseModel, responses={400: {"model": ErrorResponseModel}, 500: {"model": ErrorResponseModel}}, description="Recibe una pregunta y devuelve una respuesta y su fuente.")
async def ask(question_data: Question):
    question = question_data.question

    if not question:
        raise HTTPException(status_code=400, detail="No se proporcionó una pregunta")

    response_text, source = logic(question)
    return {'response': response_text, 'source': source}


@app.get('/execute_drive_script', response_model=List[Document], responses={500: {"model": ErrorResponseModel}}, description="Obtiene documentos de Google Drive.")
def run_drive_script():
    """
    Obtain Google Drive docs.
    """
    try:
        source_documents = execute_drive_script()
        return source_documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.get('/process_documents', response_model=List[Document], responses={500: {"model": ErrorResponseModel}})
def run_pdf_processing():
    """
    Excecute embeddings process
    ---
    tags:
      - Embeddings
    responses:
      201:
        description: Process completed
      500:
        description: Server error
    """
    try:
        df_embeddings = process_documents()
        return df_embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/classify_documents', response_model=List[Document], responses={500: {"model": ErrorResponseModel}})
def classify_downloaded_documents_route():
    """
    Clasify documents in categories: Risk Assessment, Contracts, Regulatory, Claims utilizando Langchain
    """
    try:
        classified_docs = classify_documents()
        return classified_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



