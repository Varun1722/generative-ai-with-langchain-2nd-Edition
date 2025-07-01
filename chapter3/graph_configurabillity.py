from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from IPython.display import Image,display
from typing import Literal,List,Annotated,Optional,Union
import asyncio
from operator import add
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

def my_reducer(left: List[str], right: Optional[Union[str,List[str]]]) -> List[str]:
    if right:
        return left+[right] if isinstance(right,str) else left+right
    return left

class JobApplication(TypedDict):
    Job_Description: str
    is_suitable: bool
    application: str
    # actions: List[str]
    # actions: Annotated[List[str],add]
    actions: Annotated[List[str],my_reducer]
#define functions which will act as our node

def analyse_application(state):
    print("analysing a job description")
    result = {
        "is_suitable":len(state["Job_Description"])<100,
        "actions":["action1"]}
    return result

def generating_application(state: JobApplication, config: RunnableConfig):
    model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.0-flash")
    print(f"...generating application with {model_provider} and {model_name} ...")
    return {"application":"fake application", "actions":"action2"}

def is_suitable_application(state: JobApplication) -> Literal["generating_application", END]:
    if state.get("is_suitable"):
        return "generating_application"
    return END

#build a graph and add node to it
graph = StateGraph(state_schema=JobApplication)
graph.add_node("analyse_application",analyse_application)
graph.add_node("generating_application",generating_application)

graph.add_edge(START,"analyse_application")
graph.add_conditional_edges("analyse_application",is_suitable_application)
graph.add_edge("generating_application",END)

main_graph = graph.compile()
# res = main_graph.invoke({"Job_Description":"fake_jd"})
res = main_graph.invoke({"Job_Description":"fake_jd"},config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}})
print(res)