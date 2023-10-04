from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.document_loaders import DirectoryLoader
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch


load_dotenv('.env')
embeddings = HuggingFaceHubEmbeddings()
raw_documents = DirectoryLoader('document/', glob="**/*").load()
text_splitter = CharacterTextSplitter(separator="\n",
                                    chunk_size=1000,
                                    chunk_overlap=200,
                                    length_function=len)
documents = text_splitter.split_documents(raw_documents)
# client = QdrantClient("localhost", port=6333)
# client.recreate_collection(
#         collection_name="test_collection",
#         vectors_config=VectorParams(size=768, distance=Distance.DOT),
# )

# vector_store = Qdrant(
#     client=client, 
#     collection_name="test_collection", 
#     embeddings=embeddings,
# )
# vector_store.add_documents(documents)



client = MongoClient('mongodb+srv://Cluster48666:thang4567@cluster48666.ieyo0wo.mongodb.net')

db_name = "langchain_db"
collection_name = "langchain_col"
collection = client[db_name][collection_name]
index_name = "langchain_demo"
docsearch = MongoDBAtlasVectorSearch.from_documents(
    documents, embeddings, collection=collection, index_name=index_name
)