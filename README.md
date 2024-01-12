#### 灵积-通义千问 Langchain
探索 通义千问 Api 在 langchain 中的使用
参考借鉴 openai langchain 的实现 
目前在个人项目工具中使用

NOTE: langchian 已经带有了一个合并的 `Tongyi` 实现, 当时写这个项目的时候 Tongyi 的功能还不够完善, 
不过随着后续的迭代应该已经没问题了 建议优先考虑通过以下方式使用
```py
from langchain_community.llms.tongyi import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
```

#### Install 
pip 会同时安装依赖库: Langchain 和 Dashscope-SDK
```sh
pip install langchain-qianwen
```

Clone 项目 手动安装
```sh
git clone ... && cd langchain_qianwen
pip install -r requirements.txt

 # 建议运行 pytest 单元测试确认功能运行正常，防止依赖库出现 breaking change
pip install pytest
pytest
```

使用前置条件：
1. 了解 Langchain [langchain文档](https://python.langchain.com/docs/get_started/installation)
2. 在阿里云开发参考文档 [申请并创建API-KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)
3. 设置 api_key 环境变量 `export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"`


#### 支持 LCEL(LangChain Expression Language) 语法
```py
from langchain.prompts import PromptTemplate
from langchain_qianwen import Qwen_v1

if __name__ == "__main__":
    jock_template = "给我讲个有关 {topic} 的笑话"
    prompt = PromptTemplate.from_template(jock_template)

    llm = Qwen_v1(
        model_name="qwen-turbo",
        temperature=0.18,
        streaming=True,
    )

    chain = prompt | llm

    for s in chain.stream({"topic": "产品经理"}):
        print(s, end="", flush=True)
```

#### 支持异步调用 async callback handler
p.s. 目前 llm 模型 (Qwen_v1) 可以使用 AsyncIteratorCallbackHandler, 
chatmodel(ChatQwen_v1 待更新 这个我还用不到...)
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

更多使用请查看 langchain 官方文档 和 examples 目录