import os
import bs4
from dotenv import load_dotenv
from helpers.vectore_store_utils import add_documents_to_pinecone_index, create_pinecone_index
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain import hub
load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

def initialize_rag_system():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    embedding_model = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

    qa_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # Text splitters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    index = create_pinecone_index('webloader-index')  # Create a Pinecone index with a specific name
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    retriever = vector_store.as_retriever() # Creates a Retriever object, which is a runnable. All vector stores have as_retriever() method.
        
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Prompt template for contextualizing the question based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

# Initialize the RAG system
conversational_rag_chain = initialize_rag_system()

if __name__ == "__main__":
    while True:
        user_query = input("what would you like to know? ")
        response = conversational_rag_chain.invoke(
            {"user_input": user_query},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )["answer"]
        print(response)


# Flow-
# RunnableWithMessageHistory internally invokes history_aware_retriever with user input and chat history, 
# which passes both into llm and gets a context aware question.
# The context aware question is then passed to the create_retrieval_chain, 
# which retrieves relevant documents from the vector store.
# The retrieved documents are then passed to the create_stuff_documents_chain, which invokes llm with
# the context, chat history and user query to generate an answer.