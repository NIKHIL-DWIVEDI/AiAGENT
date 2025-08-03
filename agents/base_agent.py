from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from tools.calculator import calculator
from langchain.agents import AgentExecutor,create_tool_calling_agent

class BaseAgent:
    def __init__(self,model_name='llama3.2:3b', temperature=0.1, max_tokens=1000):
                
        self.llm = ChatOllama(model=model_name,temperature=temperature,max_tokens=max_tokens)
        self.tools = [calculator]
        ## create the agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        

    def _create_agent(self):
        system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt, ("placeholder", "{agent_scratchpad}")],

        )
        ## bind tools to LLM first
        return create_tool_calling_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )
    
    def run(self, input_text):
        try:
            response = self.agent_executor.invoke({"input": input_text})
            return response
        except Exception as e:
            return f"An error occurred: {e}"
