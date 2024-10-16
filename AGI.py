import os
import re
import torch
import json
import subprocess
import time
import operator
import random
from typing import Any, List, Optional, Dict, Union, Annotated, TypedDict

from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolCall
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import run_in_executor
from langchain_core.agents import AgentAction
from langchain.tools import Tool
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver

# Custom imports
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen2.5-14B-Instruct"

try:
    print(f"Models directory: {os.environ['HF_HOME']}")
except:
    print("HF_HOME not set")
    os.environ['HF_HOME'] = '/data/placido/cache/'
    command = ['python3', 'AGI.py']
    subprocess.check_call(command)
    exit()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Define tools
@tool
def get_files_list(dir: str) -> dict:
    """Tool to list all files and directories in the directory with their types.
    The argument 'dir' specifies the directory to list."""
    try:
        files_list = os.listdir(dir)
        result = {}
        for file_name in files_list:
            file_path = os.path.join("/data/placido/AGI/", file_name)
            if os.path.isdir(file_path):
                result[file_name] = 'directory'
            else:
                result[file_name] = 'file'
        return result
    except Exception as e:
        return {"error": f"Error accessing directory: {str(e)}"}

@tool
def get_file_content(file_path: str) -> str:
    """Tool to get the content of a specific file specified in the argument.
    Cannot be used with directories."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {file_path} not found."
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

@tool
def run_command(command: str) -> str:
    """Tool to run a command in the terminal and return the output."""
    try:
        is_it_ok = input(f"Are you sure you want to run the command: {command}? (yes/no)\n")
        if is_it_ok.lower() != "yes":
            return "Command not executed"
        output = subprocess.check_output(command, shell=True, text=True)
        return output
    except Exception as e:
        return f"Error running command: {str(e)}"

@tool
def write_file(filename: str, content: str) -> str:
    """Tool to write content to a file specified in the argument.
    Avoids overwriting if the file already exists. Cannot be used with directories."""
    filepath = f"/data/placido/AGI/{filename}"
    
    if os.path.exists(filepath):
        return f"Error: File {filename} already exists. Writing aborted."
    
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Content written to file {filename}."
    except Exception as e:
        return f"Error writing to file {filename}: {str(e)}"

@tool
def get_user_feedback(prompt: str) -> str:
    """Tool to get user feedback based on the prompt provided.
    If unsure on next steps, use this tool to get user feedback."""
    user_input = input(f"{prompt}\n")

    return user_input

@tool
def tell_user(
    text: str,
):
    """Returns a natural language response to the user in the form of a string.
    The text argument is required and should be a string."""
    print(text)
    return text

@tool
def end_interaction() -> None:
    """Tool to end the program."""
    exit()

tools=[
    get_files_list,
    get_file_content,
    run_command,
    write_file,
    get_user_feedback,
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
    args: Dict[str, str] = Field(description="Parameters for the tool call. If the tool does not require any parameters, an empty dictionary should be provided.")
    id: str = Field(description="Unique identifier for the tool call.")
    type: str = "tool_call"

# Tool choice model with dynamic listing of available tools
class AIAnswer(BaseModel):
    content: str = Field(description="An explanation of why the tool was chosen. You need to provide this information to the user. REQUIRED.")
    tool_calls: ToolCall = Field(
        description="Tool call to be invoked. Only one tool call is allowed. All keys are required.",
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
    output_parser: JsonOutputParser = JsonOutputParser(pydantic_object=AIAnswer)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Handles LLM inference with video and text inputs."""
        messages_new = []
        for msg in messages:
            # print(f"\n{msg.pretty_repr()}")
            if type(msg) == SystemMessage:
                messages_new.append({"role": "system", "content": msg.content})
            elif type(msg) == HumanMessage:
                messages_new.append({"role": "user", "content": msg.content})
            elif type(msg) == AIMessage:
                messages_new.append({"role": "assistant", "content": msg.content})
        messages = messages_new
        # print(f"\n######\n{messages}\n######\n")

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

        # print(f"\n-------\n{response}\n-------\n")
        try:
            #if there is the "json" string in the response, we need to remove it
            pattern = r'^```json\s*(.*?)\s*```$'
            cleaned_string = re.sub(pattern, r'\1', response, flags=re.DOTALL)
            parsed_output = self.output_parser.parse(cleaned_string)
            # print(f"\n-------\n{parsed_output}\n-------\n")
            content = parsed_output["content"]
            tool_calls = parsed_output["tool_calls"]
            # check if is a list
            print(f"°°°°°°°°°°°°°{tool_calls}")
            if not isinstance(tool_calls, list):
                print("°°°°°°°°°°°°°NOT A LIST")
                tool_calls = [tool_calls]
            ai_message = AIMessage(content=content, tool_calls=tool_calls)
        except Exception as e:
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

        return llm

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

   - If a tool has already been used with a specific query (as noted in the scratchpad), do NOT reuse that tool for the same query.
   - Aim to gather information from a diverse array of sources before formulating your response to the user. 
   - The user's chat history will be provided first, followed by their current prompt.
   - If possible, prefer other tools over the 'run_command' tool.
   - When in doubt, use the 'get_user_feedback' tool to ask the user for more information.
   - Once you have sufficient information to answer the user's question, use the `tell_user` tool to communicate your response.

Your goal is to provide the most comprehensive and informed answer possible."""

oracle_prompt = ChatPromptTemplate.from_messages([
    ("system", llm.output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")),
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

Your task is to analyze the user's query to determine the most appropriate next steps for the oracle. Your primary goal is to assist the oracle in achieving the user's objectives efficiently.

- You have already received all the messages exchanged between the user and the oracle. Do not request additional information from the user.
- The user's chat history will be provided first.
- Based on the latest actions taken, provide a clear and concise message of steps that the oracle should take next.
- Do not specify the arguments for the tools; That's the oracle's job. 

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
    print("\n\n************************\nRUN_ORACLE\n")
    # print(f"- input: {state['input']}")
    # print(f"- chat_history: {state['chat_history']}")
    # print(f"- intermediate_steps: {state['intermediate_steps']}")
    # state_with_no_history = state.copy()
    # state_with_no_history['chat_history'] = []
    out = oracle.invoke(state)
    # check if all required fields are present, otherwise do a recursive call
    if not all(field in out.tool_calls[0] for field in ["name", "args", "id", "type"]):
        print("!!!!! NOT ALL REQUIRED FIELDS WERE PRESENT !!!!!")
        return run_oracle(state)

    print(out.content)

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
    print("\n\n************************\nRUN_REFLECTION\n")
    state["input"] = "Given the current state, what should I do next?"
    out = reflect.invoke(state)
    print(out.content)
    # time.sleep(10)
    return {
        "input": out.content,
    }


def router(state: list):
    # print("\n************************\nrouter")
    # print(state)
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
    print(f"\n************************\n{tool_name}.invoke(input={tool_args}) -> {out}\n")
    # print(f"\n************************\n{tool_name}.invoke(input={tool_args})\n")
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
    if tool_obj.name != "end_interaction":
        graph.add_edge(tool_obj.name, "reflect")

graph.add_edge("reflect", "oracle")
graph.add_edge("end_interaction", END)

runnable = graph.compile()

starting_prompt = "Give me the tree of the content of the current directory."
print("\n************************")
print(f"Starting prompt: {starting_prompt}")
print("************************\n")

out = runnable.invoke({
    "input": starting_prompt,
    "chat_history": [],
    "intermediate_steps": []
})