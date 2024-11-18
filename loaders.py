from langchain_community.document_loaders import (WebBaseLoader, YoutubeLoader, CSVLoader, PyMuPDFLoader, TextLoader) 

def carrega_site(url):
    loader = WebBaseLoader(url)
    lista_documento = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documento])
    return documento

def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    lista_documento = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documento])
    return documento

def carrega_pdf(caminho):
    loader = PyMuPDFLoader(caminho)
    lista_documento = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documento])
    return documento

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documento = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documento])
    return documento