from fastapi import FastAPI, HTTPException, APIRouter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from appfrwk.config import get_config
from operator import itemgetter
from .document_processing import _combine_documents
from .models import DocumentModel, DocumentResponse
from .store import AsnyPgVector
from .store_factory import get_vector_store
from appfrwk.logging_config import get_logger
from semantic_text_splitter import TiktokenTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
import hashlib

config = get_config()
log = get_logger(__name__)

# Router information
router = APIRouter(
    prefix="/RAG",
    tags=["RAG"],
    responses={404: {"description": "Not found"}},
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
history = []
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
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    retriever = pgvector_store.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | model
                            | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | model | StrOutputParser()

except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


def add_routes(app):
    app.include_router(router)


@router.post("/add-documents/")
async def add_documents(documents: list[DocumentModel]):
    try:

        pdf_text = documents[0].page_content
        splitter = TiktokenTextSplitter("gpt-3.5-turbo", trim_chunks=False)
        MIN_TOKENS = 100
        MAX_TOKENS = 1000

        chunks_with_model = splitter.chunks(pdf_text, chunk_capacity=(MIN_TOKENS, MAX_TOKENS))
        for i, chunk in enumerate(chunks_with_model):
            log.info(f"CHUNK WITH MODEL {i + 1}: ")
        docs = [
            Document(
                page_content=doc,
                metadata=(

                    {"digest": hashlib.md5(doc.encode()).hexdigest()}
                ),
            )
            for doc in chunks_with_model
        ]
        ids = (
            await pgvector_store.aadd_documents(docs)
        )

        return {"message": "Documents added successfully", "id": ids}

    except Exception as e:
        log.error(f"Internal error 500: {e}")
        raise HTTPException(status_code=500)


@router.get("/get-all-ids/")
async def get_all_ids():
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            ids = await pgvector_store.get_all_ids()
        else:
            ids = pgvector_store.get_all_ids()

        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-documents-by-ids/", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            documents = await pgvector_store.get_documents_by_ids(ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            documents = pgvector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-documents/")
async def delete_documents(ids: list[str]):
    try:

        existing_ids = await pgvector_store.get_all_ids()
        await pgvector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/")
async def quick_response(msg: str):
    history.append(HumanMessage(content=msg))
    try:
        result = conversational_qa_chain.invoke(
            {
                "question": msg,
                "chat_history": history,
            }
        )
        history.append(AIMessage(content=result))
        return result
    except Exception as e:
        log.error(f"error code 500 {e}")
        raise HTTPException(status_code=500, detail=str(e))
