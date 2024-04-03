from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
        callbacks=[StreamingStdOutCallbackHandler()],
        plugins={'code_interpreter': {}},  # choose the desired plugin(s).
        streaming=False,
    )

    # pylint: disable=invalid-name
    question = "使用python 画一个y=x^2的函数图"
    resp = llm(question)
    print("resp: ", resp)
