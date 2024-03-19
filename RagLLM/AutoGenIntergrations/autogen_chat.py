import autogen

from RagLLM.LangChainIntergrations.langchainlayer import LangChainService
from appfrwk.config import get_config
from user_proxy_webagent import UserProxyWebAgent
import asyncio

config = get_config()
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": config.OPENAI_API_KEY
    }
]
llm_config_assistant = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "config_list": config_list,
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
}
llm_config_proxy = {
    "model": "gpt-3.5-turbo-0613",
    "temperature": 0,
    "config_list": config_list,
}


#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat():
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()
        """You are a helpful assistant, Answer the question based on the context. 
                                                     Keep the answer accurate. Respond "Unsure about answer" if not sure about the answer."""
        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config_assistant,
            system_message="""You are a helpful assistant,, Answer the question based on the context. 
                Keep the answer accurate. Respond "Unsure about answer" if not sure about the answer.
            When you ask a question, always add the word "BRKT"" at the end.
            When you responde with the your thesis add the word TERMINATE"""
        )
        self.user_proxy = UserProxyWebAgent(
            name="user_proxy",
            human_input_mode="ALWAYS",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            function_map={
                "search_db": self.answer_PDF_question
            }
        )

        # add the queues to communicate 
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)

    async def start(self, message):
        await self.user_proxy.a_initiate_chat(
            self.assistant,
            clear_history=True,
            message=message
        )

    # MOCH Function call
    def answer_PDF_question(self, question):
        template = """Answer the question based only on the following context:
               {context}

               Question: {question}
               """
        qa = LangChainService(model_name=config.SERVICE_MODEL, template=template)

        response = qa.rag_chain.invoke({"question": question,
                                        "chat_history": qa.get_message_history(),
                                        })
        return response
