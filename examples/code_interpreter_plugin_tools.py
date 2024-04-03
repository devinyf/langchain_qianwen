from typing import List
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool

from langchain.agents.mrkl.base import ZeroShotAgent
from langchain_qianwen.tools import code_interpreter_plugin

from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-max",
    )

    tools: List[BaseTool] = [code_interpreter_plugin]

    custom_agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools)
    agent_exector = AgentExecutor.from_agent_and_tools(
        agent=custom_agent, tools=tools, verbose=True
    )

    question = "使用 python 画一个y=x^2的函数图"
    agent_exector.run(question)
