from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class AgentName(BaseModel):
    agent_name: str = "brandi"


class AgentsBase(BaseModel):
    """
    Agents base schema
    """

    context: str
    first_message: str
    response_shape: str
    instructions: str


class AgentCreate(AgentsBase):
    """
    Agent creation schema
    """
    pass


class ConversationBase(BaseModel):
    """
    Conversation base schema

    Attributes:
    - `agent_id (str)`: Agent id
    - `user_sub (str)`: User sub
    """

    user_sub: str


class ConversationCreate(ConversationBase):
    """
    Conversation Creation schema
    """
    pass


class Conversation(ConversationBase):
    """
    Conversation schema
    """
    id: str
    created_at: datetime = datetime.utcnow()

    class Config:
        from_attributes = True


class Agent(AgentsBase):
    """
    Agent Schema

    Attributes:
    - `context (str)`: Context for the AI agent.
        - "You are a chef specializing in Mediterranean food that provides receipts with a maximum of simple 10 ingredients.
        The user can have many food preferences or ingredient preferences, and your job is always to analyze and guide them
        to use simple ingredients for the recipes you suggest and these should also be Mediterranean.
        The response should include detailed information on the recipe.
        The response should also include questions to the user when necessary.
        If you think your response may be inaccurate or vague, do not write it and answer with the exact text: \`I don't have a response.\`"

    - `first_message (str)`: First message AI agent sends.
        - "Hello, I am your personal chef and cooking advisor and I am here to help you with your meal preferences and your cooking skills.
        What can I can do for you today?"

    - `response_shape (str)`: Expect response for each agent's interaction with a user (for programmatic communication).
        - "'recipes': 'List of strings with the name of the recipes',
        'ingredients': 'List of the ingredients used in the recipes',
        'summary': 'String, summary of the conversation'}"

    - `instructions (str)`: Instructions for the AI agent.
        - "Run through the conversation messages and discard any messages that are not relevant for cooking.
        Focus on extracting the recipes that were mentioned in the conversation and for each of them extract the list of ingredients.
        Make sure to provide a summary of the conversation when asked."
    - `id`: Agent id
    - `Created_at`: Agent creation date
    """
    id: str
    conversations: List[Conversation] = []
    created_at: datetime = datetime.utcnow()

    class Config:
        from_attributes = True

# Internal Schemas


class MessageBase(BaseModel):
    """
    Message base schema
    """
    user_message: str
    agent_message: str


class MessageCreate(MessageBase):
    """
    Message creation schema
    """
    pass


class Message(MessageBase):
    """
    Message schema
    """
    id: str
    conversation_id: str
    created_at: datetime = datetime.utcnow()

    class Config:
        from_attributes = True

# API Schemas


class UserMessage(BaseModel):
    """
    User message schema
    """
    conversation_id: str
    message: str


class ChatAgentResponse(BaseModel):
    """
    Chat agent response schema
    """
    conversation_id: str
    response: str