from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        )

    question = "你好， 今天天气怎么样"
    llm(question)
