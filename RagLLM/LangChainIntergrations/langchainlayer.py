from operator import itemgetter

from langchain_core.messages import get_buffer_string
from langchain_core.prompts import MessagesPlaceholder

from appfrwk.config import get_config
from appfrwk.logging_config import get_logger
from langchain.chat_models import ChatOpenAI as LChainChatOpenAI
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from RagLLM.Processing.document_processing import combine_documents
from RagLLM.PGvector.store_factory import get_vector_store

log = get_logger(__name__)
config = get_config()


class LangChainService:
    def __init__(self, template: str, model_name=None, verbose=True, streaming=True,
                 contextualize_q_system_prompt=None, qa_system_prompt=None, chat_history_template=None):
        self.model_name = model_name or config.SERVICE_MODEL
        self.verbose = verbose
        self.streaming = streaming
        self.template = template
        self.openai_api_key = config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in the configuration")

        # Use provided prompts or default to current prompts
        self.contextualize_q_system_prompt = contextualize_q_system_prompt or \
                                             """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

        self.qa_system_prompt = qa_system_prompt or \
                                """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\
            {context}"""

        self.chat_history_template = chat_history_template or \
                                     """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. \
            Chat History: \
            {chat_history} \
            Follow Up Input: {question} \
            Standalone question:"""

        self._initialize_memory_and_parser()
        self._initialize_llm()
        self._initialize_retriever_and_templates()
        self._initialize_chains()

    # Rest of the class implementation...

    def _initialize_memory_and_parser(self):
        """Initialize memory management and output parser."""
        self.history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(chat_memory=self.history, return_messages=config.DEBUG)
        self.str_output_parser = StrOutputParser()

    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            self.llm = LChainChatOpenAI(
                model_name=self.model_name,
                temperature=config.SERVICE_TEMPERATURE,
                max_tokens=config.SERVICE_MAX_TOKENS,
                streaming=self.streaming,
                verbose=self.verbose,
                openai_api_key=self.openai_api_key,
                model_kwargs={"frequency_penalty": config.SERVICE_FREQUENCY_PENALTY,
                              "presence_penalty": config.SERVICE_PRESENCE_PENALTY}
            )
        except Exception as e:
            log.error(f"Failed to initialize language model: {e}")
            raise

    def _initialize_retriever_and_templates(self):
        """Initialize the retriever and templates for conversational retrieval."""
        connection_string = config.DATABASE_URL2
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        mode = "async"
        self.pgvector_store = get_vector_store(
            connection_string=connection_string,
            embeddings=embeddings,
            collection_name="testcollection",
            mode=mode,
        )
        self.retriever = self.pgvector_store.as_retriever()
        self._initialize_templates()

    def _initialize_templates(self):
        """Centralized template initialization."""

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.chat_history_template)
        self.ANSWER_PROMPT = ChatPromptTemplate.from_template(self.template)
        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _initialize_chains(self):
        """Simplify method calls for initializing various chains."""
        self._initialize_conversational_qa_chain()
        self._initialize_contextualize_q_chain()
        self._initialize_rag_chain()

    # Additional methods for chain initializations...
    def _initialize_conversational_qa_chain(self):
        """Create the conversational QA chain with the retriever."""

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
                                | self.CONDENSE_QUESTION_PROMPT
                                | self.llm
                                | StrOutputParser(),
        )
        _context = {
            "context": itemgetter("standalone_question") | self.retriever | combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        self.conversational_qa_chain = _inputs | _context | self.ANSWER_PROMPT | self.llm | StrOutputParser()

    def _initialize_contextualize_q_chain(self):

        self.contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()

    def _initialize_rag_chain(self):
        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return self.contextualize_q_chain
            else:
                return input["question"]

        format_docs = lambda docs: " ".join([doc["content"] for doc in docs])  # Define or import format_docs as needed

        self.rag_chain = (
                RunnablePassthrough.assign(
                    context=contextualized_question | self.retriever | format_docs
                )
                | self.qa_prompt
                | self.llm
        )

    def get_message_history(self):
        """Return the message history."""
        return self.history.messages

    def add_user_message(self, message):
        """Add a user message to the history."""
        self.history.add_user_message(message)

    def add_ai_message(self, message):
        """Add an AI-generated message to the history."""
        self.history.add_ai_message(message)
