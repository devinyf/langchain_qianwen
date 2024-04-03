# Import things that are needed generically
from langchain.tools import tool
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import Qwen_v1


@tool
def code_interpreter_plugin(query: str) -> str:
    """use it when you need to solve the problem with python code.
       the action input should not be the final code, this tool will generate code itself and figure out the answer.
    """
    llm = Qwen_v1(
        model_name="qwen-turbo",
        # model_name="qwen-plus",
        plugins={'code_interpreter': {}},  # choose the desired plugin(s).
    )

    if query.startswith("```py"):
        query = "execute the code below to get result: \n" + query

    resp = llm(query)
    return resp
