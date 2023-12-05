import json

from dashscope.api_entities.dashscope_response import GenerationResponse

JSON_STRING = """
{
    "status_code": 200,
    "request_id": "bb74f872-cdf9-93c8-b0fa-170a2045db34",
    "code": "",
    "message": "",
    "output": {
        "text": null,
        "finish_reason": null,
        "choices": [{
            "finish_reason": "null",
            "message": {
                "role": "assistant",
                "content": "%s"
            }
        }]
    },
    "usage": {
        "input_tokens": 1,
        "output_tokens": 8,
        "total_tokens": 9
    }
}
"""

MOCK_CALL_RESPONSE = 'hello, Im a mock-data response!'
MOCK_STREAMING_RESPONSE = 'hello, Im a mock-data stream!'


def mock_call_response():
    return GenerationResponse(**json.loads(JSON_STRING % (MOCK_CALL_RESPONSE)))


def mock_streaming_generator():
    tmpstr = ""
    for _, v in enumerate(MOCK_STREAMING_RESPONSE):
        tmpstr += v
        resp = GenerationResponse(**json.loads(JSON_STRING % (tmpstr)))
        if v == "?":
            resp.output.choices[0].finish_reason = "stop"

        yield resp

