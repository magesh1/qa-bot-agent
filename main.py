import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from operator import itemgetter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain import hub

# hide deprecate warning
# import warning
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# define vector db
chroma_db_dir = "./chroma_db"


# load file
def fileLoader(path) -> list[Document]:
    documents = []
    loader = PyPDFLoader(path)
    try:
        documents.extend(loader.load())
    except Exception as e:
        print(f"error loading {path}: {e}")
        if not os.path.exists(path):
            print(f"File not found: {path}")
    return documents


# text splitting to fit tokens
def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
        split document into smaller chunks
    :param documents: pdf file
    :param chunk_size: 1000
    :param chunk_overlap: 200
    :return:
    """
    print(f"\nSplitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(documents)
    print(f"Original documents: {len(documents)}, Split into {len(split_documents)} chunks.")
    return split_documents


# vector embeddings (convert text to numeric and fits with dimension)
def get_ollama_embeddings(model_name: str = "mistral:7b"):
    """
        Initializes and returns an OllamaEmbeddings object.
    """
    print(f"\nInitializing Ollama Embeddings with model: {model_name}...")
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        test_embedding = embeddings.embed_query("hello world")
        print(f"Successfully initialized embeddings. Embedding dimension: {len(test_embedding)}")
        return embeddings
    except Exception as e:
        print(f"Error initializing Ollama Embeddings with model {model_name}: {e}")
        print(
            "Please ensure Ollama is running and you have pulled the specified model (e.g., 'ollama pull llama3').")
        return None


# create or load from chroma db
def create_or_load_chroma_vector_store(documents: list[Document], embeddings: OllamaEmbeddings,
                                       collection_name: str = "my_rag_collection"):
    """
    Creates a Chroma vector store from documents or loads an existing one.
    """
    if os.path.exists(chroma_db_dir) and len(os.listdir(chroma_db_dir)) > 0:
        print(f"\nAttempting to load Chroma Vector Store from '{chroma_db_dir}'...")
        try:
            vectorstore = Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            if vectorstore._collection.count() > 0:
                print("Chroma Vector Store loaded successfully with existing data.")
                return vectorstore
            else:
                print("Chroma directory exists but collection is empty or inaccessible. Recreating...")
        except Exception as e:
            print(f"Error loading existing Chroma vector store: {e}. Recreating...")

    print(f"\nCreating new Chroma Vector Store in '{chroma_db_dir}'...")
    try:
        os.makedirs(chroma_db_dir, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_dir,
            collection_name=collection_name
        )
        vectorstore.persist()
        print("New Chroma Vector Store created and persisted.")
        return vectorstore
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        return None


# used to retrieve relevant answer from document
def setup_rag_chain(vectorstore, ollama_model_name: str = "mistral:7b"):
    """
       Sets up the Retrieval Augmented Generation (RAG) chain.
    """
    print(f"\nSetting up RAG chain with Ollama model: {ollama_model_name}...")
    llm = ChatOllama(model=ollama_model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt_template = ChatPromptTemplate.from_template("""
       You are an AI assistant for question-answering tasks.
       Use the following retrieved context to answer the question.
       If you don't know the answer, just say that sorry it was not in my memory, don't try to make up an answer.
       Keep the answer concise and to the point.

       Context: {context}
       Question: {question}

       Answer:
       """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": itemgetter("question") | retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )
    print("RAG chain set up successfully.")
    return rag_chain


# setup websearch tool
def get_web_search_tool():
    """Initializes and returns a web search tool."""
    print("\nInitializing Web Search Tool (DuckDuckGo)...")
    search_tool = DuckDuckGoSearchRun()
    return search_tool


# creating agent for rag and web search
def setup_agent(vectorstore, ollama_model_name: str = "mistral:7b"):
    """
        Sets up a hybrid Q&A agent that can use both RAG and web search.
    """
    print(f"\nSetting up Hybrid Q&A Agent with Ollama model: {ollama_model_name}...")
    agent_llm = ChatOllama(model=ollama_model_name, temperature=0.1)

    rag_chain_tool_executor = setup_rag_chain(vectorstore, ollama_model_name)

    def rag_tool_func(query: str):
        """Wrapper function for RAG tool that handles the input properly"""
        try:
            result = rag_chain_tool_executor.invoke({"question": query})
            return str(result)
        except Exception as e:
            return f"Error querying documents: {str(e)}"

    rag_tool = Tool(
        name="document_qa_tool",
        description="""
              Useful for answering questions that can be found in the provided local documents.
              The input to this tool should be a plain string, representing the question to search for in the documents.
              For example: "What is the capital of France?" or "Explain quantum mechanics."
              """,
        func=rag_tool_func,
    )

    def web_search_func(query: str):
        """Wrapper function for web search tool"""
        try:
            web_search = get_web_search_tool()
            result = web_search.run(query)
            return str(result)
        except Exception as e:
            return f"Error searching web: {str(e)}"

    web_search_tool = Tool(
        name="web_search_tool",
        description="""
                   Useful for searching the web for current information, general knowledge,
                   or anything not likely to be in the provided documents.
                   The input to this tool should be a plain string, representing the search query.
                   For example: "current weather in London" or "history of the internet."
                   """,
        func=web_search_func,
    )

    tools = [rag_tool, web_search_tool]

    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub

        # Get the standard ReAct prompt
        prompt = hub.pull("hwchase17/react")

        # Create agent with the standard approach
        agent = create_react_agent(agent_llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
        )

    except Exception as e:
        print(f"Failed to create ReAct agent: {e}")
        print("Falling back to initialize_agent method...")

        # Fallback: Use the old method with a very explicit prompt
        agent_executor = initialize_agent(
            tools=tools,
            llm=agent_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="force"
        )

    print("Hybrid Q&A Agent set up successfully.")
    return agent_executor


def run_chat_loop(qa_chain):
    """
    Runs the chat loop with proper error handling
    """
    print("\n--- Document Q&A Bot Ready ---")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if qa_chain:
            try:
                response = qa_chain.invoke({"input": user_input})
                if isinstance(response, dict) and 'output' in response:
                    print(f"Bot Answer: {response['output']}")
                else:
                    print(f"Bot Answer: {str(response)}")

            except Exception as e:
                print(f"Error in chat loop: {e}")
                print("Let me try a different approach (direct web search fallback)..")
                try:
                    web_search = get_web_search_tool()
                    result = web_search.run(user_input)
                    print(f"Web Search Result: {result}")
                except Exception as e2:
                    print(f"Fallback web search also failed: {e2}")
        else:
            print("Q&A chain not initialized. Cannot answer questions.")


if __name__ == "__main__":
    raw_documents = fileLoader("facts.pdf")
    if not raw_documents:
        print("No documents loaded. Exiting.")
    else:
        document_chunk = split_documents(raw_documents)
        embeddings = get_ollama_embeddings()
        if not embeddings:
            print("Failed to initialize embeddings. Exiting.")
        else:
            vector_store = create_or_load_chroma_vector_store(document_chunk, embeddings)
            if not vector_store:
                print("Failed to create/load Chroma DB. Exiting.")
            else:
                qa_chain = setup_agent(vector_store)
                if not qa_chain:
                    print("Failed to setup agent. Exiting.")
                else:
                    run_chat_loop(qa_chain)
