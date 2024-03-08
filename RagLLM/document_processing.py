from fastapi import HTTPException
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .store_factory import get_vector_store
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document
config = get_config()
log = get_logger(__name__)
import hashlib


async def process_and_store_documents(documents, splitter):
    try:
        # Splitting the document into chunks
        pdf_text = documents[0].page_content
        chunks = split_document(pdf_text, splitter)

        # Preparing documents for storage
        docs = prepare_documents_for_storage(chunks)

        # Storing documents
        ids = await store_documents(docs)
        return {"message": "Documents added successfully", "id": ids}
    except Exception as e:
        log.error(f"Internal error 500: {e}")
        raise HTTPException(status_code=500)


def split_document(text, splitter):
    MIN_TOKENS = 100
    MAX_TOKENS = 1000
    return splitter.chunks(text, chunk_capacity=(MIN_TOKENS, MAX_TOKENS))


def prepare_documents_for_storage(chunks: list[str]):
    return [
        Document(page_content=chunk, metadata={"digest": hashlib.md5(chunk.encode()).hexdigest()})
        for chunk in chunks
    ]

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
async def store_documents(docs):
    # This should interface with your actual storage solution, e.g., pgvector_store
    # The implementation would look something like this:
    try:
        CONNECTION_STRING = f"postgresql+psycopg2://myuser:mypassword@db:5432/mydatabase"

        OPENAI_API_KEY = config.OPENAI_API_KEY
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        mode = "async"
        pgvector_store = get_vector_store(
            connection_string=CONNECTION_STRING,
            embeddings=embeddings,
            collection_name="testcollection",
            mode=mode,
        )
        return await pgvector_store.add_documents(docs)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
