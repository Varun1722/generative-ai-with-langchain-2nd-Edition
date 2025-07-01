from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from IPython.display import Image,display
from typing import Literal,List,Annotated,Optional,Union
import asyncio
from operator import add
from langchain_core.runnables.config import RunnableConfig
from enum import Enum
from langchain.output_parsers import EnumOutputParser
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

class isSuitableEnum(Enum):
    YES = "YES"
    NO = "NO"

parser = EnumOutputParser(enum=isSuitableEnum)
assert parser.invoke("NO") == isSuitableEnum.NO 
assert parser.invoke("YES\n") == isSuitableEnum.YES 
assert parser.invoke(" YES \n") == isSuitableEnum.YES
assert parser.invoke(HumanMessage(content="YES")) == isSuitableEnum.YES


job_description = """
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH
Neuhausen-Hamberg
Feste Anstellung
Homeoffice möglich, Vollzeit
Erschienen: vor 1 Tag
Glaston Germany GmbH logo
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH"""
prompt_template_enum = (
    "Given a job description, decide whether it suites a junior Java developer."
    "\nJOB DESCRIPTION:\n{job_description}\n\nAnswer only YES or NO."
)


def my_reducer(left: List[str], right: Optional[Union[str,List[str]]]) -> List[str]:
    if right:
        return left+[right] if isinstance(right,str) else left+right
    return left

class JobApplication(TypedDict):
    Job_Description: str
    is_suitable: isSuitableEnum
    application: str
    # actions: List[str]
    # actions: Annotated[List[str],add]
    actions: Annotated[List[str],my_reducer]
#define functions which will act as our node

chain = llm|parser
result = chain.invoke(prompt_template_enum.format(job_description=job_description))

def analyse_application(state):
    job_description = state["Job_Description"]
    prompt = prompt_template_enum.format(job_description=job_description)
    result = chain.invoke(prompt)
    return {"is_suitable": result}

def generating_application(state: JobApplication, config: RunnableConfig):
    model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.0-flash")
    print(f"...generating application with {model_provider} and {model_name} ...")
    return {"application":"fake application", "actions":"action2"}

def is_suitable_application(state: JobApplication) -> Literal["generating_application", END]:
    return state['is_suitable']==isSuitableEnum.YES

#build a graph and add node to it
graph = StateGraph(state_schema=JobApplication)
graph.add_node("analyse_application",analyse_application)
graph.add_node("generating_application",generating_application)

graph.add_edge(START,"analyse_application")
graph.add_conditional_edges("analyse_application",is_suitable_application,
                            {True:"generating_application",False:END})
graph.add_edge("generating_application",END)

main_graph = graph.compile()
# res = main_graph.invoke({"Job_Description":"fake_jd"})
res = main_graph.invoke({"Job_Description":"fake_jd"})
print(res)