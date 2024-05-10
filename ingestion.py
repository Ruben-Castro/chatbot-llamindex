import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore


# load the environment variables
load()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


if __name__ == "__main__":
    print("going to ingest documentation into pinecode vector db")
    UnstructuredReader = download_loader("UnstructuredReader")

    # point to the directory we want to scan and use the unstructured reader to parse throught
    # the directory and use it on html documents. in the file_extractor dictionary we can specify
    # other extractors for pdf documents, docx, powerpoints etc.
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs", file_extractor={".html": UnstructuredReader()}
    )

    documents = dir_reader.load_data()

    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # temp = 0 means no creative ans
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )
    # convert nodes into vector embeddings

    # save embedings into a vector database
    index_name = "llama-index-documentation-helper"
    pinecone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    print("...finished embeding")
