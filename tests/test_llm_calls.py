from unittest.mock import patch
from unittest import TestCase
import sys

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from mock_data import (mock_call_response, mock_streaming_generator,
                       MOCK_CALL_RESPONSE, MOCK_STREAMING_RESPONSE)

from langchain_qianwen import Qwen_v1

# 添加路径到sys.path
LOCAL_PACKAGE_PATH = '../langchain_qianwen/'
if LOCAL_PACKAGE_PATH not in sys.path:
    sys.path.append(LOCAL_PACKAGE_PATH)


class TestQwenLLM(TestCase):
    def test_llm(self):
        llm = Qwen_v1(
            model_name="qwen-turbo",
            streaming=False,
        )

        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_call_response()

            question = "你好"
            response = llm(question)
            assert response == MOCK_CALL_RESPONSE

    def test_llm_stream(self):

        llm = Qwen_v1(
            model_name="qwen-turbo",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],

        )
        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_streaming_generator()
            question = "你好"
            response = llm(question)

            assert response == MOCK_STREAMING_RESPONSE
