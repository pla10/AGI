import os
import torch
import random
from typing import Any, List, Optional, Dict
from typing import Annotated, Literal, TypedDict
from typing import AsyncIterator, Iterator

from langchain_core.language_models.llms import LLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor

from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
0
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import ToolCall, ToolMessage


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Define tools
@tool
def get_files_list() -> List[str]:
    """Tool to list all files in the directory."""
    try:
        return os.listdir("/data/placido/MA-LMM/AGI")
    except Exception as e:
        return [f"Error accessing directory: {str(e)}"]

@tool
def get_file_content(file: str) -> str:
    """Tool to get the content of a specific file specified in the argument."""
    try:
        with open(f"/data/placido/MA-LMM/AGI/{file}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {file} not found."
    except Exception as e:
        return f"Error reading file {file}: {str(e)}"

@tool
def final_answer(
    text: str,
):
    """Returns a natural language response to the user in the form of a string."""
    return text

@tool
def end_interaction() -> None:
    """Tool to end the program."""
    exit()

tools=[
    get_files_list,
    get_file_content,
    final_answer,
    end_interaction
]

# print(get_files_list.invoke(input={}))
# print(get_file_content.invoke(input={"file": "AGI2.py"}))
# print(end_interaction.invoke(input={}))
# print("HELLO")

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define a model for tool call parameters
class ToolCall(BaseModel):
    name: str = Field(description="Name of the tool to be invoked.")
    args: Dict[str, str] = Field(default={}, description="Parameters for the tool call. This field is required.")
    id: str = Field(description="Unique identifier for the tool call.")
    type: str = "tool_call"

# Tool choice model with dynamic listing of available tools
class AIAnswer(BaseModel):
    # content: str = Field(description="Content of the message. This should specify next steps.")
    tool_calls: ToolCall = Field(
        description="Tool call to be invoked. Only one tool call is allowed.",
        title="Tool Call",
        examples=[
            {
                "name": tool.name,
                "args": tool.args,
                "id": str(random.randint(0, 1000)),
                "type": "tool_call",
            }
            for tool in tools]
    )

# QwenLLM class with tool binding and invocation
class QwenLLM(BaseChatModel):
    model: Any
    processor: Any
    tools: Dict[str, callable] = {}
    output_parser: Optional[JsonOutputParser] = None  # Add output_parser attribute

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Handles LLM inference with video and text inputs."""
        print(f"\n++++++\n{messages}\n++++++\n")

        formatted_messages = [{"role": "user", "content": [{"type": "text", "text": msg.content}] } for msg in messages]
        messages = formatted_messages

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Model inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse the output if an output parser is set
        print(f"\n-------\n{output_text}\n-------\n")
        if self.output_parser:
            parsed_output = self.output_parser.parse(output_text[0])
            print(f"\n-------\n{parsed_output}\n-------\n")
            tool_calls = parsed_output["tool_calls"]
            ai_message = AIMessage(content="", tool_calls=tool_calls)
        else:
            ai_message = AIMessage(content=output_text[0])

        # Return the AIMessage wrapped in ChatGeneration and ChatResult
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "custom"


    def bind_tools(self, tools: List[Tool]) -> LLM:
        """Bind a list of tools (functions) to the LLM instance."""
        llm = self
        llm.tools = {tool.name: tool for tool in tools}
        llm.output_parser = JsonOutputParser(pydantic_object=AIAnswer)
        # Set up parser and prompt template
        prompt = PromptTemplate(
            template="{format_instructions}\n{input}",
            input_variables=["input"],
            partial_variables={"format_instructions": llm.output_parser.get_format_instructions()},
        )

        # Combine prompt and model into a chain
        chain = prompt | llm
        return chain


# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Initialize the LLM and bind tools
llm = QwenLLM(model=model, processor=processor)

system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])


# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    agent_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            agent_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(agent_steps)

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools)
)

inputs = {
    "input": "tell me something interesting about dogs",
    "chat_history": [],
    "intermediate_steps": [],
}

out = oracle.invoke(inputs)
print([out])