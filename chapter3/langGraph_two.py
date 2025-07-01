from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from IPython.display import Image,display
from typing import Literal

# load_dotenv()

class JobApplication(TypedDict):
    Job_Description: str
    is_suitable: bool
    application: str

#define functions which will act as our node
def analyse_application(state):
    print("analysing a job description")
    return {"is_suitable":len(state["Job_Description"])>100}

def generating_application(state):
    print("generating suitable application")
    return {"application":"fake application"}

def is_suitable_application(state: JobApplication) -> Literal["generating_application", END]:
    if state.get("is_suitable"):
        return "generating_application"
    return END

#build a graph and add node to it
graph = StateGraph(JobApplication)
graph.add_node("analyse_application",analyse_application)
graph.add_node("generating_application",generating_application)

graph.add_edge(START,"analyse_application")
graph.add_conditional_edges("analyse_application",is_suitable_application)
graph.add_edge("generating_application",END)

main_graph = graph.compile()
# result = main_graph.invoke({"Job_Description":"fake_jd"})
# print(result)

display(Image(main_graph.get_graph().draw_mermaid_png()))
with open("langgraph_workflow_2.png",'wb') as f:
    f.write(main_graph.get_graph().draw_mermaid_png())