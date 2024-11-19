import streamlit as st
import tempfile
import os
import tiktoken
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (WebBaseLoader, YoutubeLoader, CSVLoader, PyMuPDFLoader, TextLoader) 
from dotenv import load_dotenv
from loaders import *
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]

if api_key is None:
    raise ValueError("A chave da API não foi encontrada. Verifique o arquivo .env.")
else:
    print("Chave da API carregada com sucesso.")

TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Txt']
MEMORIA = ConversationBufferMemory()
ARQUIVO_CONTROLE_TOKENS = 'consumo_tokens.txt'

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
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
        Seu nome é Ina, você é a Inteligência Artificial da Opuspac University, que é um braço acadêmico da empresa Opuspac. Você é uma garota inteligente, delicada, simpática, proativa e assertiva.
        Você possui acesso às seguintes informações vindas de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário apertar CTRL F5!'''.format(tipo_arquivo, documento)
    
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        (MessagesPlaceholder(variable_name="chat_history")),
        ('user', "{input}")
    ])
    
    modelo = 'gpt-3.5-turbo'
    chat = ChatOpenAI(model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain

def salvar_tokens_txt(caminho_arquivo, contagem_tokens):
    with open(caminho_arquivo, 'a') as arquivo:
        arquivo.write(str(contagem_tokens) + '\n')

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

    input_usuario = st.chat_input('Mande uma mensagem...')
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
        
        # Salvar a contagem de tokens no arquivo TXT
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
    elif tipo_arquivo == 'Youtube':
        arquivo = st.sidebar.text_input('Digite a URL do vídeo')
    elif tipo_arquivo == 'Pdf':
        arquivo = st.sidebar.file_uploader('Faça o upload do arquivo PDF', type=['pdf'])
    elif tipo_arquivo == 'Txt':
        arquivo = st.sidebar.file_uploader('Faça o upload do arquivo TXT', type=['txt'])

    carregar = st.sidebar.button("Carregar Arquivo")
    if st.sidebar.button("Limpar Conversa"):
        st.session_state.pop('memoria', None)
    return tipo_arquivo, arquivo, carregar

def main():
    tipo_arquivo, arquivo, carregar = sidebar()  
    if carregar and tipo_arquivo and arquivo:
        carrega_modelo(api_key, tipo_arquivo, arquivo)

    pagina_chat()

if __name__ == '__main__':
    main()
