def classify_documents():
    import json
    from langchain.chains import create_tagging_chain
    from langchain_openai import ChatOpenAI
    import pandas as pd


    classified_docs = []
    # Lee el archivo JSON con los títulos y URLs de los documentos
    with open('source_documents.json') as file:
        documents = json.load(file)

    # Schema
    schema = {
        "properties": {
            "category": {
                "type": "string",
                "enum": ["Risk Assessment", "Contracts", "Regulatory", "Claims"],
                "description": "The category of the document",
            },
            "title": {
                "type": "string",
            },
            "url": {
                "type": "string",
            }
        },
        "required": ["category", "url", "title"],
    }

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    chain = create_tagging_chain(schema, llm)

    # Itera a través de cada documento
    for title, url in documents.items():
        # Aquí podrías agregar el código para extraer el texto del documento desde el URL si es necesario
        # Por ahora, asumiremos que solo pasas el título al modelo
        inp = title
        result = chain.invoke(inp)

        classified_docs.append({
                        'title': title,
                        'tag': result['text']['category'],
                        'source': url
                    })
        
    # Convertir la lista de documentos clasificados a un DataFrame
    df_classified = pd.DataFrame(classified_docs)

    # Opcionalmente, guarda los resultados clasificados en un archivo CSV
    df_classified.to_csv('./classified_documents.csv', index=False)

    return classified_docs

