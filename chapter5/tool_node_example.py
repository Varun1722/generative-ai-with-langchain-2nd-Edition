from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent, ToolNode,tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

@tool
def get_date(year: int, month: int = 1, day: int = 1) -> date:
    """Returns a date object given year, month and day.

      Default month and day are 1 (January) and 1.
      Examples in YYYY-MM-DD format:
        2023-07-27 -> date(2023, 7, 27)
        2022-12-15 -> date(2022, 12, 15)
        March 2022 -> date(2022, 3)
        2021 -> date(2021)
    """
    return date(year, month, day).isoformat()

@tool
def time_difference(days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> date:
    """Returns a date given a difference in days, weeks, months and years relative to the current date.

    By default, days, weeks, months and years are 0.
    Examples:
      two weeks ago -> time_difference(weeks=2)
      last year -> time_difference(years=1)
    """
        # Step 1: Subtract days and weeks
    dt = date.today() - timedelta(days=days, weeks=weeks)

    # Step 2: Subtract months safely
    total_months = dt.month - months
    year_offset = total_months // 12
    new_month = total_months % 12 or 12
    new_year = dt.year - years + (year_offset if total_months > 0 else -1)

    # Handle last day of month edge cases
    try:
        result_date = dt.replace(year=new_year, month=new_month)
    except ValueError:
        # If day does not exist in new month (e.g. Feb 30), fallback to last day of month
        result_date = dt.replace(year=new_year, month=new_month, day=1)
        next_month = result_date.replace(day=28) + timedelta(days=4)
        result_date = next_month - timedelta(days=next_month.day)

    return result_date.isoformat()

examples = [
  "I signed my contract 2 years ago",
  "I started the deal with your company in February last year",
  "Our contract started on March 24th two years ago"
]

agent = create_react_agent(
    llm, [get_date, time_difference], prompt="Extract the starting date of a contract. Current year is 2025.")


for example in examples:
  result = agent.invoke({"messages": [("user", example)]})
  print(example, result["messages"][-1].content)