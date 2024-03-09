from typing import List, AsyncIterable

from RagLLM.LangChainIntergrations.LangChainLayer import LangChainService
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