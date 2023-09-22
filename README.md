#### 阿里云 灵积 通义千问 Langchain
通义千问的 langchain llms 和 chat_models
参考借鉴 openai langchain 的实现 
目前仅作为个人学习 langchain 使用
持续更新中...

#### llms
```py
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tongyi_qwen_langchain import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-v1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        )

    question = "你好， 今天天气怎么样"
    llm(question)

```

#### chat_models
```py
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tongyi_qwen_langchain import ChatQwen_v1
from langchain.schema import (
    HumanMessage,
)

if __name__ == "__main__":
    chat = ChatQwen_v1(
        model_name="qwen-v1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    chat([HumanMessage(content="介绍一下 Golang")])
```
#### 使用 agent 增加网络查询功能
```py
from langchain.agents import load_tools, AgentType, initialize_agent
from tongyi_qwen_langchain import Qwen_v1

if __name__ == "__main__":
    llm = Qwen_v1(
        ##经测试只有 plus 模型才能正常使用 agent
        model_name="qwen-plus",
    )
    ## 需要去 serpapi 官网申请一个 api_key
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)

    agent = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    agent.run("深圳今天出门用带雨伞吗?")
```