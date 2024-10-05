import openai
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(".env", override=True)
llm = ChatOpenAI()

class Chatbot:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        with open("prompts/system.txt") as f:
            sys_prompt = f.read()
        self.system_prompt = SystemMessage(content=sys_prompt)

    def fmt_message(self, message, exchange=None, classification=None) -> HumanMessage:
        if exchange and classification and exchange != "" and classification != "":
            # I could use langchain's prompt templates, which I have done before, but not really needed.
            with open("prompts/with_exchange.txt") as f:
                prompt = f.read()
            prompt = prompt.replace("<<exchange>>", exchange)
            prompt = prompt.replace("<<classification>>", classification)
            prompt = prompt.replace("<<user_message>>", message)
            user_message = prompt
        else:
            user_message = message
        return HumanMessage(content=user_message)

    def get_response(self, messages) -> str:
        # note that we need to add the system prompt every time because it is not stored in the DB.
        # This also lets us dynamically change it without having to update the database. in my experience this is very
        # helpful.
        response = llm.invoke([self.system_prompt] + messages)
        return response.content