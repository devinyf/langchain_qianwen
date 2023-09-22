# flake8: noqa

PREFIX = """|你是一个可以智能助理, 必须通过提供的工具(tools)来完成任务, 请严格按照下面要求的格式返回相关的答案:
现在有以下工具(Tools):"""

FORMAT_INSTRUCTIONS = """
对话会以 Question 作为开始, 例如:
Question: 用户输入的问题

你回复的格式必须严格符合下列两种模版(Template)中的其中一个(Action or Final):
```Action Template
Thought: 思考如何寻找合适的方法，工具解决问题
Action: 接下来要执行的动作，可执行的动作列表为 [{tool_names}]
Action Input: 执行动作需要的输入参数
```
```Final Template
Thought: 我找到了答案
Final Answer: 用户输入问题的最终答案
```
...(一系列的 Thought/Action/Action Input/Observation 可能会重复多次)
"""

SUFFIX = """现在开始

Question: {input}
Thought:{agent_scratchpad}"""
