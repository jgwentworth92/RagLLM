import hashlib
from typing import List
import asyncio
import autogen
from fastapi import HTTPException, APIRouter, Depends, WebSocket
from langchain.globals import set_debug
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from semantic_text_splitter import TextSplitter
from RagLLM.AutoGenIntergrations import AutoGenService
from RagLLM.AutoGenIntergrations.autogen_chat import AutogenChat
from RagLLM.LangChainIntergrations.langchainlayer import LangChainService
from RagLLM.PGvector.models import DocumentModel, DocumentResponse
from RagLLM.PGvector.store import AsnyPgVector
from RagLLM.PGvector.store_factory import get_vector_store
from RagLLM.Processing.langchain_processing import load_conversation_history
from RagLLM.database import agent_schemas as schemas
from RagLLM.database import db, crud, agent_schemas
from RagLLM.database.user_schemas import UserCreate

from appfrwk.config import get_config
from appfrwk.logging_config import get_logger

set_debug(True)
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
template = """Answer the question based only on the following context:
   {context}

   Question: {question}
   """

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
    db.connect()

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
        splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", trim_chunks=False)
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


@router.post("/create-conversation", response_model=schemas.Conversation)
async def create_conversation(conversation: schemas.ConversationCreate,
                              db_session=Depends(db.get_db)) -> schemas.Conversation:
    """ Create conversation """

    try:

        user_sub = conversation.user_sub
        user = await crud.get_user_by_sub(db_session, user_sub)
        if not user:
            log.info(f"Sub not found, creating user")
            user = await crud.create_user(db_session, UserCreate(sub=user_sub))
        log.info(f"User retrieved")
        log.info(f"Creating conversation")
        db_conversation = await crud.create_conversation(db_session, conversation)
        log.info(f"Conversation created")
        return db_conversation
    except Exception as e:
        log.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rag_chain_chat/")
async def quick_response(message: schemas.UserMessage, db_session=Depends(db.get_db)):
    Service = LangChainService(model_name=config.SERVICE_MODEL, template=template)

    try:
        conversation = await crud.get_conversation(db_session, message.conversation_id)
        log.info(f"User Message: {message.message}")

    except Exception as e:
        log.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    try:
        chathistory = load_conversation_history(conversation, Service)
        log.info(f"current chat history {Service.get_message_history()}")

        result = Service.rag_chain.invoke(
            {
                "question": message.message,
                "chat_history": Service.get_message_history(),
            }
        )

        db_messages = agent_schemas.MessageCreate(
            user_message=message.message, agent_message=result, conversation_id=conversation.id)
        await crud.create_conversation_message(db_session, message=db_messages, conversation_id=conversation.id)
        return result
    except Exception as e:
        log.error(f"error code 500 {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/autogen_rag_chain_chat/")
async def autogen_rag_response(message: schemas.UserMessage):
    config_list = [
        {
            "model": "gpt-4-1106-preview",
            "api_key": config.OPENAI_API_KEY,
        }
    ]

    Service = AutoGenService(config_list)
    try:
        logging_session_id = autogen.ChatCompletion.start_logging()
        result = await Service.user_proxy.initiate_chat(
            Service.assistant,
            message=message.message,
            clear_history=True,
        )

        log.info("Logging session ID: " + str(logging_session_id))
        return result

    except Exception as e:
        log.error(f"error code 500 {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-conversations", response_model=List[schemas.Conversation])
async def get_user_conversations(user_sub: str, db_session=Depends(db.get_db)) -> List[schemas.Conversation]:
    """
    Get all conversations for an agent by id
    """
    try:
        log.info(f"Getting all conversations for user sub: {user_sub}")
        db_conversations = await crud.get_user_conversations(db_session, user_sub)
        return db_conversations
    except Exception as e:
        log.error(f"Error retrieving conversations for user_sub: {user_sub}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/get-conversation-messages", response_model=List[schemas.MessageCreate])
async def get_conversation_messages(conversation_id: str, db_session=Depends(db.get_db)) -> List[schemas.MessageCreate]:
    """
    Get all messages for a conversation by id
    """
    try:
        log.info(
            f"Getting all messages for conversation id: {conversation_id}")
        db_messages = await crud.get_conversation_messages(db_session, conversation_id)
        return db_messages
    except Exception as e:
        log.error(
            f"Error retrieving messages for conversation id: {conversation_id}")
        log.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[AutogenChat] = []

    async def connect(self, autogen_chat: AutogenChat):
        await autogen_chat.websocket.accept()
        self.active_connections.append(autogen_chat)

    async def disconnect(self, autogen_chat: AutogenChat):
        autogen_chat.client_receive_queue.put_nowait("DO_FINISH")
        print(f"autogen_chat {autogen_chat.chat_id} disconnected")
        self.active_connections.remove(autogen_chat)


manager = ConnectionManager()


async def send_to_client(autogen_chat: AutogenChat):
    while True:
        reply = await autogen_chat.client_receive_queue.get()
        if reply and reply == "DO_FINISH":
            autogen_chat.client_receive_queue.task_done()
            break
        await autogen_chat.websocket.send_text(reply)
        autogen_chat.client_receive_queue.task_done()
        await asyncio.sleep(0.05)


async def receive_from_client(autogen_chat: AutogenChat):
    while True:
        data = await autogen_chat.websocket.receive_text()
        if data and data == "DO_FINISH":
            await autogen_chat.client_receive_queue.put("DO_FINISH")
            await autogen_chat.client_sent_queue.put("DO_FINISH")
            break
        await autogen_chat.client_sent_queue.put(data)
        await asyncio.sleep(0.05)


@router.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket,message: schemas.UserMessage):
    try:
        autogen_chat = AutogenChat(chat_id=message.conversation_id, websocket=websocket)
        await manager.connect(autogen_chat)
        data = await autogen_chat.websocket.receive_text()
        future_calls = asyncio.gather(send_to_client(autogen_chat), receive_from_client(autogen_chat))
        await autogen_chat.start(data)
        print("DO_FINISHED")
    except Exception as e:
        print("ERROR", str(e))
    finally:
        try:
            await manager.disconnect(autogen_chat)
        except:
            pass
