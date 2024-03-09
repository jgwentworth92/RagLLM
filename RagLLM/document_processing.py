from typing import List, AsyncIterable

from fastapi import HTTPException
from langchain.schema import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document

import hashlib

from RagLLM.LangChainIntergrations.LangChainLayer import LangChainService
from RagLLM.database import models, crud, agent_schemas
from appfrwk.logging_config import get_logger

log = get_logger(__name__)
"""
config = get_config()
log = get_logger(__name__)
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

"""


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


def sort_message_history(conversation: models.Conversation) -> List[models.Message]:
    """
    Sorts the message history for the conversation by created_at timestamp in ascending order.

    Args:
        conversation (schemas.Conversation): The conversation to sort the message history for.
    """
    message_history = conversation.messages
    message_history.sort(key=lambda x: x.created_at, reverse=False)
    return message_history


def load_conversation_history(conversation: models.Conversation, service: LangChainService):
    """
    Loads the conversation history into the LangChainService.

    Args:
        conversation: The conversation model from the database.
        service: The LangChainService instance.
    """
    try:
        # Load initial agent message if the conversation is new
        ai_first_msg = conversation.agent.first_message
        service.add_ai_message(ai_first_msg)

        # Load existing conversation messages
        for msg in sort_message_history(conversation):
            service.add_user_message(msg.user_message)
            service.add_ai_message(msg.agent_message)

    except Exception as e:
        log.error(f"Error loading conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def generate_streaming_response(db_session, service: LangChainService, user_message: str,
                                      conversation: models.Conversation) -> AsyncIterable[str]:
    """
    Generates a streaming response from the LangChainService.

    Args:
        db_session: The database session.
        service: The LangChainService instance.
        user_message: The user message.
        conversation: The conversation model from the database.
    """
    # Initialize an empty response
    response = ""

    try:
        # Process the stream
        async for result in service.conversational_qa_chain.astream({"question": user_message,
                                                                     "chat_history": conversation.messages}):
            # Add the result to the response
            response += result

            # Log the result
            log.info(f"Stream: {result}")

            # Yield the result
            yield result
    except Exception as e:
        log.error(f"An error occurred in generate_streaming_response: {str(e)}")
        response = "Sorry, I'm having technical difficulties."
        yield response
    finally:
        # Save the complete response to the database
        db_messages = agent_schemas.MessageCreate(
            user_message=user_message, agent_message=response, conversation_id=conversation.id)
        await crud.create_conversation_message(db_session, message=db_messages, conversation_id=conversation.id)


"""
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
"""
