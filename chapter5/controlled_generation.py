from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

class Step(BaseModel):
    """ A step of the workflow"""
    step: str = Field(description="Description of the step")

class Plan(BaseModel):
    """A step by step plan to execute a task"""
    steps: list[Step]

prompt = PromptTemplate.from_template(
    "Prepare a step-by-step plan to execute the given task"
    "Task\n{task}\n"
)

result = (prompt | llm.with_structured_output(Plan)).invoke(
    "How to write a bestseller on Amazon about Rust?"
)

assert isinstance(result, Plan)
print(f"No of steps: {len(result.steps)}")
for steps in result.steps:
    print(steps.step)
    break
