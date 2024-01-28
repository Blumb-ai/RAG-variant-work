from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./documentos/20190528_Reglament_Hospitalitzacio_web.pdf")
pages = loader.load_and_split()

print(pages[0])

