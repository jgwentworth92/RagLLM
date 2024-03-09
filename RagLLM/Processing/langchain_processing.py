from typing import List, AsyncIterable

from fastapi import HTTPException

from RagLLM.LangChainIntergrations.langchainlayer import LangChainService
from RagLLM.database import agent_schemas, crud, models
from appfrwk.logging_config import get_logger

log = get_logger(__name__)


def sort_message_history(conversation: models.Conversation) -> List[models.Message]:
    """
    Sorts the message history for the conversation by created_at timestamp in ascending order.

    Args:
        conversation (schemas.Conversation): The conversation to sort the message history for.
    """
    message_history = conversation.messages
    message_history.sort(key=lambda x: x.created_at, reverse=False)
    return message_history


def load_conversation_history(conversation: models.Conversation, service):
    """
    Loads the conversation history into the LangChainService, ensuring that chat_history
    is always a list.

    Args:
        conversation: The conversation model from the database.
        service: The LangChainService instance.
    """
    try:

        if conversation and conversation.messages:
            log.info("conversation has messages")
            # Load existing conversation messages
            for msg in sort_message_history(conversation):
                service.add_user_message(msg.user_message)
                service.add_ai_message(msg.agent_message)
        else:
            log.info("new conversation")
            service.add_ai_message("hi how may i help you")

        # Now chat_history is guaranteed to be a list, though it could be empty
        # Here, instead of directly adding messages to the service, you would
        # appropriately pass `chat_history` where required, ensuring it's never None

    except Exception as e:
        log.error(f"Error loading conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def generate_streaming_response(db_session, service, user_message: str,
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
