import os
import re
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

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import ToolCall, ToolMessage

from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ['HF_HOME'] = '/data/placido/cache/'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# Define tools
@tool
def get_files_list() -> List[str]:
    """Tool to list all files in the directory."""
    try:
        return os.listdir("/data/placido/AGI/")
    except Exception as e:
        return [f"Error accessing directory: {str(e)}"]

@tool
def get_file_content(file: str) -> str:
    """Tool to get the content of a specific file specified in the argument."""
    try:
        with open(f"/data/placido/AGI/{file}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {file} not found."
    except Exception as e:
        return f"Error reading file {file}: {str(e)}"

@tool
def tell_user(
    text: str,
):
    """Returns a natural language response to the user in the form of a string."""
    print(text)
    return text

@tool
def end_interaction() -> None:
    """Tool to end the program."""
    exit()

tools=[
    get_files_list,
    get_file_content,
    tell_user,
    end_interaction
]

# print(get_files_list.invoke(input={}))
# print(get_file_content.invoke(input={"file": "AGI2.py"}))
# print(end_interaction.invoke(input={}))
# print("HELLO")

class AgentState(TypedDict):
    input: str
    chat_history: Annotated[list[tuple[BaseMessage, str]], operator.add]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define a model for tool call parameters
class ToolCall(BaseModel):
    name: str = Field(description="Name of the tool to be invoked.")
    args: Dict[str, str] = Field(default={}, description="Parameters for the tool call. This field is required.")
    id: str = Field(description="Unique identifier for the tool call.")
    type: str = "tool_call"

# Tool choice model with dynamic listing of available tools
class AIAnswer(BaseModel):
    # content: str = Field(description="Content of the message. This should specify next steps.")
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
    tokenizer: Any
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
        print(f"\n*******\n{messages}\n*******\n")

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": messages[0].content}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"\n-------\n{response}\n-------\n")

        # Parse the output if an output parser is set
        if self.output_parser:
            try:
                #if there is the "json" string in the response, we need to remove it
                pattern = r'^```json\s*(.*?)\s*```$'
                cleaned_string = re.sub(pattern, r'\1', response, flags=re.DOTALL)
                parsed_output = self.output_parser.parse(cleaned_string)
                print(f"\n-------\n{parsed_output}\n-------\n")
                tool_calls = parsed_output["tool_calls"]
                ai_message = AIMessage(content="", tool_calls=tool_calls)
            except Exception as e:
                ai_message = AIMessage(content=response)
        else:
            ai_message = AIMessage(content=response)

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

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the LLM and bind tools
llm = QwenLLM(model=model, tokenizer=tokenizer)

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

system_prompt = """You are the Oracle, a powerful AI decision-maker.

Your task is to determine the appropriate action for each user query based on the available tools.

1. **Tool Usage Limitations**:
   - If a tool has already been used with a specific query (as noted in the scratchpad), do NOT reuse that tool for the same query.
   - No tool may be used more than twice. If a tool has appeared in the scratchpad twice, it cannot be used again.

2. **Information Gathering**:
   - Aim to gather information from a diverse array of sources before formulating your response to the user. 
   - Store the collected information in the scratchpad.

3. **Responding to Users**:
   - The user's chat history will be provided first, followed by their current prompt. 
   - Once you have sufficient information to answer the user's question, use the `tell_user` tool to communicate your response.

Your goal is to provide the most comprehensive and informed answer possible."""

oracle_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | oracle_prompt
    | llm.bind_tools(tools)
)

system_prompt1 = f"""You are a highly effective reflecting agent, dedicated to providing the best suggestions for the oracle.

Your task is to analyze the user's query and the complete chat history to determine the most appropriate next steps for the oracle. Your primary goal is to assist the oracle in achieving the user's objectives efficiently.

1. **Context Analysis**:
   - The user's chat history will be provided first. Summarize the relevant parts of the chat history that pertain to the user's request.

2. **Response Crafting**:
   - Based on the latest actions taken, provide a clear and concise message that the oracle can convey to the user. 

3. **Next Action Suggestion**:
   - In this case, suggest that the oracle communicates the retrieved list of files ('AGI.py' and '.git') to the user using the `tell_user` tool. Make it explicit that this is the next action the oracle should take.

4. **Tool Awareness**:
   - Be mindful of the available tools at your disposal and mention them if they are relevant to your suggestions.

List of available tools: {str([tool.name for tool in tools])}
"""


reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt1),
    MessagesPlaceholder(variable_name="chat_history"),
    ("ai", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

reflect = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | reflect_prompt
    | llm
)


def run_oracle(state: list):
    print("\n************\nrun_oracle")
    print(f"input: {state['input']}")
    print(f"chat_history: {state['chat_history']}")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    # state_with_no_history = state.copy()
    # state_with_no_history['chat_history'] = []
    out = oracle.invoke(state)
    print(f"out: {out}")
    print(f"out.tool_calls: {out.tool_calls}")
    if out.tool_calls:
        tool_name = out.tool_calls[0]["name"]
        tool_args = out.tool_calls[0]["args"]
    else:
        return {
            "chat_history": [state["input"]],
            "intermediate_steps": []
        }  # Handle the case when no tool is suggested

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "chat_history": [state["input"]],
        "intermediate_steps": [action_out]
    }

def run_reflection(state: list):
    print("\n************\nrun_reflection")
    state["input"] = "Given the current state, what should I do next?"
    out = reflect.invoke(state)
    print(f"out: {out}")
    return {
        "input": out.content,
    }


def router(state: list):
    print("\n************\nrouter")
    print(state)
    if state["intermediate_steps"] == []:
        return "oracle"
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "tell_user"

tool_str_to_func = {tool.name: tool for tool in tools}

def run_tool(state: list):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    print(f"\n************\n{tool_name}.invoke(input={tool_args}) -> {out}\n")
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {
        "intermediate_steps": [action_out]
    }

graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
for tool_obj in tools:
    graph.add_node(tool_obj.name, run_tool)
graph.add_node("reflect", run_reflection)

graph.set_entry_point("oracle")

graph.add_conditional_edges(
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)

# create edges from each tool back to the oracle
for tool_obj in tools:
    if tool_obj.name != "tell_user":
        graph.add_edge(tool_obj.name, "reflect")

graph.add_edge("reflect", "oracle")

# if anything goes to final answer, it must then move to END
graph.add_edge("tell_user", END)

runnable = graph.compile()

out = runnable.invoke({
    "input": "Get list of files.",
    "chat_history": [],
    "intermediate_steps": []
})