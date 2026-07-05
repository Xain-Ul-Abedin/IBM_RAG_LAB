import os
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation (RAG) with LangChain & IBM Granite/Ollama")
    parser.add_argument(
        "--provider",
        choices=["ollama", "replicate", "watsonx"],
        default="ollama",
        help="LLM provider to use (default: ollama). Note: replicate and watsonx require API keys."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What did the president say about Ketanji Brown Jackson?",
        help="Query to ask the RAG system."
    )
    return parser.parse_args()

def setup_embeddings():
    print("[*] Setting up HuggingFace Embeddings model: ibm-granite/granite-embedding-small-english-r2...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer

    embeddings_model_path = "ibm-granite/granite-embedding-small-english-r2"
    
    # Initialize embeddings model (downloads model weights to cache on first run)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_path)
    
    # Initialize tokenizer for the text splitter
    try:
        tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
    except Exception as e:
        print(f"[!] Warning: Could not load HuggingFace tokenizer: {e}. Falling back to default tokenization.")
        tokenizer = None
        
    return embeddings, tokenizer

def load_and_split_document(tokenizer):
    print("[*] Downloading and loading the State of the Union document...")
    import wget
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

    filename = 'state_of_the_union.txt'
    url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

    if not os.path.isfile(filename):
        print(f"[*] Downloading {filename}...")
        wget.download(url, out=filename)
        print() # New line after download progress bar

    loader = TextLoader(filename, encoding="utf-8")
    documents = loader.load()

    if tokenizer:
        print("[*] Splitting document into chunks using HuggingFace tokenizer...")
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=tokenizer.max_len_single_sentence,
            chunk_overlap=0,
        )
    else:
        print("[*] Splitting document using RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    texts = text_splitter.split_documents(documents)
    
    # Add document ID metadata
    for i, text in enumerate(texts):
        text.metadata["doc_id"] = i + 1
        
    print(f"[+] Created {len(texts)} document chunks.")
    return texts

def setup_vector_db(texts, embeddings):
    print("[*] Initializing local Chroma database...")
    from langchain_chroma import Chroma
    
    # Initialize vector store
    vector_db = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Check if DB already populated
    existing_docs = vector_db.get()
    if existing_docs and len(existing_docs.get("documents", [])) > 0:
        print("[+] Vector database already populated. Reusing existing collection.")
    else:
        print("[*] Populating vector database (this might take a moment)...")
        vector_db.add_documents(texts)
        print("[+] Vector database populated successfully.")
        
    return vector_db

def get_llm(provider):
    if provider == "replicate":
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            print("[!] Error: REPLICATE_API_TOKEN environment variable not set.")
            print("[!] Please set it in your .env file or export it.")
            sys.exit(1)
        
        print("[*] Connecting to Replicate (Granite 4.1 8B)...")
        from langchain_replicate import ChatReplicate
        model_path = "ibm-granite/granite-4.1-8b"
        return ChatReplicate(
            model=model_path,
            replicate_api_token=token,
        )

    elif provider == "watsonx":
        apikey = os.getenv("WATSONX_APIKEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        
        if not apikey or not project_id:
            print("[!] Error: WATSONX_APIKEY or WATSONX_PROJECT_ID environment variable not set.")
            print("[!] Please follow Task 1-3 in README.md and update your .env file.")
            sys.exit(1)
            
        print("[*] Connecting to IBM watsonx.ai (Granite 13B Chat)...")
        from langchain_ibm import ChatWatsonx
        from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
        
        parameters = TextChatParameters(
            temperature=0.7,
            max_tokens=500
        )
        return ChatWatsonx(
            model_id="ibm/granite-13b-chat-v2",
            url=url,
            apikey=apikey,
            project_id=project_id,
            parameters=parameters
        )

    elif provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        
        print(f"[*] Connecting to local Ollama server at {host} using model '{model_name}'...")
        from langchain_community.chat_models import ChatOllama
        
        # Test connection to Ollama
        import urllib.request
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        except Exception as e:
            print(f"[!] Error: Could not connect to Ollama server at {host}: {e}")
            print("[!] Is the Ollama desktop app running?")
            print("[!] If you wanted to use an API provider, pass --provider watsonx or --provider replicate")
            sys.exit(1)
            
        return ChatOllama(
            base_url=host,
            model=model_name
        )

def assemble_rag_chain(llm, retriever):
    print("[*] Assembling retrieval-augmented generation (RAG) pipeline...")
    from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    
    # Prompt template for the model - injecting both context and query input
    prompt_template = ChatPromptTemplate.from_template(
        "You are an AI assistant helping to answer questions using a set of documents.\n"
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    
    # Build combine documents chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Build RAG chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )
    return rag_chain

def main():
    args = parse_args()
    print(f"=== IBM Granite RAG Pipeline ===")
    print(f"Provider: {args.provider.upper()}")
    print(f"================================\n")
    
    # 1. Setup Embeddings & Tokenizer
    embeddings, tokenizer = setup_embeddings()
    
    # 2. Load & Split Document
    texts = load_and_split_document(tokenizer)
    
    # 3. Setup Vector Store
    vector_db = setup_vector_db(texts, embeddings)
    
    # 4. Get LLM Model
    llm = get_llm(args.provider)
    
    # 5. Assemble RAG Chain
    rag_chain = assemble_rag_chain(llm, vector_db.as_retriever(search_kwargs={"k": 3}))
    
    # 6. Run query
    print(f"\n[*] Query: \"{args.query}\"")
    print("[*] Running RAG retrieval and generation...")
    try:
        response = rag_chain.invoke({"input": args.query})
        print("\n=== Answer ===")
        print(response.get("answer", "No answer generated."))
        print("==============\n")
        
        print("[*] Retrieved contexts:")
        for idx, doc in enumerate(response.get("context", [])):
            print(f"\n[{idx+1}] Doc ID: {doc.metadata.get('doc_id')}")
            print(doc.page_content[:200] + "...")
    except Exception as e:
        print(f"[!] Error executing RAG query: {e}")

if __name__ == "__main__":
    main()
