import autogen
from langchain_community.embeddings import OpenAIEmbeddings
from RagLLM.LangChainIntergrations.langchainlayer import LangChainService
from RagLLM.PGvector.store_factory import get_vector_store
from appfrwk.config import get_config
from appfrwk.logging_config import get_logger

log = get_logger(__name__)
config = get_config()

config_list = [
    {
        "model": "gpt-4",
    }
]



def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


class AutoGenService:
    def __init__(self, config_list):
        self.llm_config = {
            "timeout": 60,
            "temperature": 0,
            "config_list": config_list,
        }
        self.config_list = config_list
        self.openai_api_key = config.OPENAI_API_KEY
        self._initialize_llm()
        self._initialize_retriever()
        self._initialize_retrieval_chain()
        self._initialize_llm_config_assistant()
        self._initialize_UserProxyAgent()

    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            self.llm_config_proxy = {
                "temperature": 0,
                "config_list": self.config_list,
                "request_timeout": 600,
                "use_docker": False
            }
        except Exception as e:
            log.error(f"Failed to initialize language model: {e}")
            raise

    def _initialize_retriever(self):
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

    def _initialize_retrieval_chain(self):
        template = """Answer the question based only on the following context:
           {context}

           Question: {question}
           """

        self.qa = LangChainService(model_name=config.SERVICE_MODEL, template=template)

    def _initialize_llm_config_assistant(self):
        llm_config_assistant = {
            "functions": [
                {
                    "name": "answer_question",
                    "description": "You can answer questions about the content of a thesis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question about the content of a thesis.",
                            }
                        },
                        "required": ["question"],
                    },
                },
            ],
            "config_list": self.config_list,

            "timeout": 120,
        }
        self.assistant = autogen.AssistantAgent(
            name="assistant",

            llm_config=llm_config_assistant,
            system_message="""You are a helpful assistant, Answer the question based on the context. 
                                             Keep the answer accurate. Respond "Unsure about answer" if not sure about the answer."""

        )

    def answer_PDF_question(self, question):
        response = self.qa.rag_chain.invoke({"question": question,
                                             "chat_history": self.qa.get_message_history(),
                                             })
        return response

    def _initialize_UserProxyAgent(self):
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
            Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
            function_map={
                "answer_PDF_question": self.answer_PDF_question
            }
        )
