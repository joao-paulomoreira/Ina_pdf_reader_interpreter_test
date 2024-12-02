import streamlit as st
import tempfile
import os
import tiktoken
import requests
import base64 
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (WebBaseLoader, CSVLoader, PyMuPDFLoader, TextLoader) 
from dotenv import load_dotenv
from loaders import *
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]

if api_key is None:
    raise ValueError("A chave da API não foi encontrada. Verifique o arquivo .env.")
else:
    print("Chave da API carregada com sucesso.")

TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Pdf', 'Txt']
MEMORIA = ConversationBufferMemory()
FILE_PATH = 'consumo_tokens.txt'
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO_OWNER = 'joao-paulomoreira'
REPO_NAME = 'Ina_pdf_reader_interpreter_test'

def estilo_modelo():
    st.set_page_config(
        page_title='Ina - Leitora de PDFs',
        page_icon='📄',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.markdown(
        """
            <style>
                .stButton > button{
                    font-size: 1em;
                }
            </style>
        """,
    unsafe_allow_html=True
    )

def obter_conteudo_atual():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        dados = response.json()
        sha = dados["sha"]
        conteudo = base64.b64decode(dados["content"]).decode("utf-8")
        return conteudo, sha
    elif response.status_code == 404:
        return "", None
    else:
        raise Exception(f"Erro ao obter conteúdo: {response.status_code}")

def atualizar_arquivo(novo_conteudo):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    conteudo_atual, sha = obter_conteudo_atual()

    conteudo_atual += novo_conteudo + "\n"
    conteudo_base64 = base64.b64encode(conteudo_atual.encode("utf-8")).decode("utf-8")

    data = {
        "message": "Atualizando consumo de tokens",
        "content": conteudo_base64,
        "sha": sha
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code in [200, 201]:
        print("Arquivo atualizado com sucesso!")
    else:
        raise Exception(f"Erro ao atualizar arquivo: {response.status_code} {response.text}")

def salvar_tokens_github(contagem_tokens):
    novo_conteudo = str(contagem_tokens) 
    atualizar_arquivo(novo_conteudo)

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    elif tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    elif tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento

def carrega_modelo(api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    system_message = '''
        Seu nome é Ina, você é a Inteligência Artificial da Opuspac University.
        Seu objetivo é resumir o conteúdo fornecido de forma extremamente concisa, com o menor número de tokens possível.
        Utilize apenas informações essenciais do documento {}: 

    ####
    {}
    ####

    Ao gerar respostas ou resumos, use uma linguagem objetiva, técnica e clara. 
    Se o conteúdo parecer não ser relevante (ex.: "Enable JavaScript..."), avise o usuário para tentar um novo upload.
    '''.format(tipo_arquivo, documento)
    
    modelo = 'gpt-3.5-turbo'
    chat = ChatOpenAI(model=modelo, api_key=api_key)
    prompt_resumo = [
    SystemMessage(content=system_message),
    HumanMessage(content='Resuma o conteúdo com o menor número de tokens possível.')
    ]
    resposta_resumo = chat.generate(messages=prompt_resumo)
    resposta_resumo = chat.generate(prompt_resumo)

    enc = tiktoken.get_encoding("cl100k_base")
    tokens_resumo = len(enc.encode(resposta_resumo))

    print(f"Resumo gerado com {tokens_resumo} tokens.")
    st.write("Resumo do Documento:")
    st.text(resposta_resumo)

    CAMINHO_ARQUIVO_TOKENS = "consumo_tokens.txt"
    salvar_tokens_txt(CAMINHO_ARQUIVO_TOKENS, tokens_resumo)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        (MessagesPlaceholder(variable_name="chat_history")),
        ('user', "{input}")
    ])
    chain = template | chat
    st.session_state['chain'] = chain


def salvar_tokens_txt(caminho_arquivo, contagem_tokens):
    with open(caminho_arquivo, 'a') as arquivo:
        arquivo.write(str(contagem_tokens) + '\n')
    
    salvar_tokens_github(contagem_tokens)

def pagina_chat():
    st.header('Opus IA - PDF', divider=True)
    chain = st.session_state.get('chain')
    
    if chain is None:
        st.error('Por favor, carregue um arquivo primeiro!')
        st.stop()
    
    memoria = st.session_state.get('memoria', MEMORIA)
    
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Pergunte sobre o resumo...')
    if input_usuario:
        user_id = st.session_state.get('user_id', 'default_user')

        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages,
            'user_id': user_id 
        }))

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(resposta)
        contagem_tokens = len(tokens)
        print(f'A quantidade de tokens usada foi {contagem_tokens}')
        
        CAMINHO_ARQUIVO_TOKENS = "consumo_tokens.txt"
        salvar_tokens_txt(CAMINHO_ARQUIVO_TOKENS, contagem_tokens)
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

        
def sidebar():
    st.sidebar.header("Upload de Arquivos")
    tipo_arquivo = st.sidebar.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
    arquivo = None

    if tipo_arquivo == 'Site':
        arquivo = st.sidebar.text_input('Digite a URL do site')
    elif tipo_arquivo == 'Pdf':
        arquivo = st.sidebar.file_uploader('Faça o upload do arquivo PDF', type=['pdf'])
    elif tipo_arquivo == 'Txt':
        arquivo = st.sidebar.file_uploader('Faça o upload do arquivo TXT', type=['txt'])

    carregar = st.sidebar.button("Carregar Arquivo")
    if st.sidebar.button("Limpar Conversa"):
        st.session_state.pop('memoria', None)
    return tipo_arquivo, arquivo, carregar

def main():
    st.session_state.pop('memoria', None)
    estilo_modelo()

    tipo_arquivo, arquivo, carregar = sidebar()

    if carregar and tipo_arquivo and arquivo:
        carrega_modelo(api_key, tipo_arquivo, arquivo)

    pagina_chat()

if __name__ == '__main__':
    main()