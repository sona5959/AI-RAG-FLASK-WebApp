from flask import Flask, Request, url_for, Response, render_template,request
from langchain_community.document_loaders.pdf import PyMuPDFLoader, PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import send2trash
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

app=Flask(__name__)

# CHUNKING  PROCESS....

def split_docs(document, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs= text_splitter.split_documents(document)
    return chunked_docs

# EMBEDDING PROCESS.....

class EmbeddinManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name=model_name
        print(f"model name.... {self.model_name}")
        self.model=SentenceTransformer(self.model_name)
        # print(f"model loaded with {self.model.get_embedding_dimension()}")

    def  generate_embeddings(self, text):
        embeddings=self.model.encode(text, show_progress_bar=True)
        print(f"embedding shape...{embeddings.shape}")
        return embeddings

# VECTOR PROCESS .....
class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documents"):
        self.persist_directory=persist_directory
        self.collection_name=collection_name
        self.collection=None
        self.client=None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client=chromadb.PersistentClient(path=self.persist_directory)
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Vector Store collection for pdf RAG"}
        )

        print(f"initialised collection with vector database {self.collection_name}")
        print(f"docs in collection {self.collection.count}")

    def add_documents(self, documents, embeddings):
        if (len(documents)!=len(embeddings)):
            raise ValueError("documents and embeddings must have same length")

        #storing data now ......from

        ids=[]
        all_metadata=[]
        documents_content=[]
        embeddings_list=[]

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id=f"doc_{uuid.uuid4()}"
            ids.append(doc_id)
            metadata=dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content_length"]=len(doc.page_content)
            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
                ids=ids,
                metadatas=all_metadata,
                embeddings=embeddings_list,
                documents=documents_content
            )

        print(f"total document added in vector store {len(documents_content)}")
        print(f"docs in collection {self.collection.count()}")


class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager=embedding_manager
        self.vector_store=vector_store

    def retrieve(self, query, top_k=3, score_threshold=0.1):
       query_embeddings =self.embedding_manager.generate_embeddings([query])[0]
       results=self.vector_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k
        )
       retrieved_docs=[]
       if results["documents"] and results["documents"][0]:
            ids=results["ids"][0]
            metadatas=results["metadatas"][0]
            documents=results["documents"][0]
            distances=results["distances"][0]

            for i, (doc_id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
                similarity_score= 1-distance
                if similarity_score>=score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "document": document,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i+1
                    })

            print(f"length of retrieved docs {len(retrieved_docs)}")

       else:
            print("no results")

       return retrieved_docs


#LLM MAKING PROCESS with LANGCHAIN....

llm = ChatGroq(groq_api_key=GROQ_API_KEY,
               model="qwen/qwen3-32b",
               temperature=0.1,
               max_tokens=1024)

def generate_output(query, rag_retriever, llm, top_k=2):
    results=rag_retriever.retrieve(query, top_k)
    context="\n".join([doc["document"] for doc in results]) if results else ""
    if not context:
        return "no results found"
    else:
        prompt= f""" use given context to generate answer for the query
        Context: {context}
        Query: {query}"""
        response=llm.invoke([prompt.format(context=context, query=query)])
        return response.content




@app.route('/')
def index():
    return render_template('index.html')

data_dir=os.path.abspath("./data")
os.makedirs(data_dir, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method=='POST':
        pdf_file=request.files.get('pdf_file')

        if pdf_file =='':
            return "Select your PDF file"
        if pdf_file and pdf_file.filename.endswith(('.pdf','.PDF')):
            pdf_file_path=os.path.join(data_dir, os.path.basename(pdf_file.filename))
            pdf_file.save(pdf_file_path)
            try:
                pdf_loader=PyMuPDFLoader(pdf_file_path)
                document=pdf_loader.load()
                chunks=split_docs(document)

                texts = [doc.page_content for doc in chunks]
                embedding_manager=EmbeddinManager()
                vector_store = VectorStoreManager()
                embedding=embedding_manager.generate_embeddings(texts)
                # print(embedding)
                vector_store.add_documents(chunks, embedding)

                rag_retriever=RAGRetriever(embedding_manager, vector_store)
                query=request.form.get("query")
                answer=generate_output(query, rag_retriever, llm)
                return answer

            finally:
                if os.path.exists(pdf_file_path):
                    os.remove(pdf_file_path)


    return "invalid file type"


@app.route('/clean')
def clean():
  folder_path = Path("data/vector_store")
  if folder_path.exists() and folder_path.is_dir():
       send2trash.send2trash(folder_path)
       return render_template('index.html')
  else:
       return "Try again for finishing chat"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)