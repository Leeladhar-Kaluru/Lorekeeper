from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

class GroqService:

    def __init__(
        self,
        model_name="openai/gpt-oss-20b"
    ):
        self.llm = ChatGroq(
            model_name = model_name,
            temperature=0
        )

    def generate(self,prompt:str)-> str:

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )
        
        return response.content