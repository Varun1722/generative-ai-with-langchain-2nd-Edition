from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from IPython.display import Image,display
from typing import Literal,List,Annotated,Optional,Union
import asyncio
from operator import add
# load_dotenv()

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

def generating_application(state):
    print("generating suitable application")
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
# result = main_graph.invoke({"Job_Description":"fake_jd"})
# print(result)

# display(Image(main_graph.get_graph().draw_mermaid_png()))
# with open("reducer_default.png",'wb') as f:
#     f.write(main_graph.get_graph().draw_mermaid_png())

async def run_stream():

    async for chunk in main_graph.astream(
        input={"Job_Description":"fake_jd"},
        stream_mode="values"
    ):
        print("Chunk received")
        print(chunk)
        print("\n")

if __name__ == "__main__":
    asyncio.run(run_stream())
