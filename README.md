#### 灵积-通义千问 Langchain
探索 通义千问 Api 在 langchain 中的使用
参考借鉴 openai langchain 的实现 
目前仅用于个人学习

前置条件：
1. 安装 langchain [langchain文档](https://python.langchain.com/docs/get_started/installation)
2. 在阿里云 [开通 DashScope 并创建API-KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)
3. 设置 api_key 环境变量 `export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"`

```
pip install langchain-qianwen
```

#### 使用 async callback handler
p.s. 目前仅 llm (Qwen_v1) 模型可以使用 AsyncIteratorCallbackHandler, chatmodel(ChatQwen_v1) 还未做更新
```py
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_qianwen import Qwen_v1
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import asyncio


async def use_async_handler(input):
    handler = AsyncIteratorCallbackHandler()
    llm = Qwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[handler], 
    )

    memory = ConversationBufferMemory()
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    asyncio.create_task(chain.apredict(input=input))

    return handler.aiter()


async def async_test():
    async_gen = await use_async_handler("hello")
    async for i in async_gen:
        print(i)


if __name__ == "__main__":
    asyncio.run(async_test())
```

#### chat_models
```py
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import ChatQwen_v1
from langchain.schema import (
    HumanMessage,
)

if __name__ == "__main__":
    chat = ChatQwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    chat([HumanMessage(content="举例说明一下 PHP 为什么是世界上最好的语言")])
```

#### 使用 agent 增加网络搜索功能
```py
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain_qianwen import Qwen_v1

if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-plus",
    )
    ## 需要去 serpapi 官网申请一个 api_key
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)

    agent = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    agent.run("今天北京的天气怎么样?")
```

#### 使用 embedding 提取文档中的信息
```py
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

from langchain_qianwen import Qwen_v1

if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
    )
    loader = DirectoryLoader("./assets", glob="**/*.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
    )

    print(f"text length: {len(texts)}")

    # 使用 embedding engion 将 text 转换为向量
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    query = "文章中的工厂模式使用例子有哪些??"
    rsp = qa.run({"query": query})
    print(rsp)

```

更多使用案例请查看 examples 目录