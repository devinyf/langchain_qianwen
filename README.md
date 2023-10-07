#### 灵积-通义千问 Langchain
探索 通义千问 在 langchain 中的使用
参考借鉴 openai langchain 的实现 
目前仅用于个人学习
持续更新中...

#### llms
```py
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-v1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        )

    question = "你好, 帮忙解释一下 Hello World 是什么意思"
    llm(question)
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
        model_name="qwen-v1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    chat([HumanMessage(content="举例说明一下 PHP 为什么是世界上最好的语言")])
```

#### 使用 agent 增加网络查询功能
```py
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain_qianwen import Qwen_v1

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
    agent.run("最近在福岛海滩上发现哥斯拉了吗?")
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
    query = "文中的工厂模式使用例子有哪些??"
    rsp = qa.run({"query": query})
    print(rsp)

```