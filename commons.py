from langchain.callbacks.manager import (
        CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
    )
from langchain.llms.base import create_base_retry_decorator

# from .qwen_llm import Qwen_v1
# from .qwen_chat_model import ChatQwen_v1
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from typing import Optional, Union, Callable, Any


def completion_with_retry(
    llm_model: BaseLLM | BaseChatModel,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm_model, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        # _kwargs["stream"] = True

        return llm_model.client.call(**_kwargs)

    return _completion_with_retry(**kwargs)


def _create_retry_decorator(
    llm_model: BaseLLM | BaseChatModel,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import dashscope

    errors = [
        # TODO: add more errors
        dashscope.common.error.RequestFailure,
        dashscope.common.error.InvalidInput,
        dashscope.common.error.ModelRequired,
    ]
    
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm_model.max_retries, run_manager=run_manager
    )
