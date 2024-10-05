import openai
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(".env", override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI()


class Chatbot:
    """
    Abstracted class for using a LLM. In this case, we're using GPT-4o-mini. This also helps me work with prompts.
    """

    @classmethod
    def fmt_message(cls, message, exchange=None, classification=None) -> HumanMessage:
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

    @classmethod
    def get_response(cls, messages, background="") -> str:
        with open("prompts/system.txt") as f:
            sys_prompt = f.read()
        sys_prompt = sys_prompt.replace("<<background>>", background)

        # note that we need to add the system prompt every time because it is not stored in the DB.
        # This also lets us dynamically change it without having to update the database. in my experience this is very
        # helpful.
        response = llm.invoke([SystemMessage(content=sys_prompt)] + messages)
        return response.content