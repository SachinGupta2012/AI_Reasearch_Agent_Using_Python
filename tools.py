from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.tools import StructuredTool
from datetime import datetime
from pydantic import BaseModel, Field


# --- Save tool ---
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = StructuredTool.from_function(
    func=save_to_txt,
    name="save_text_to_file",
    description="Saves structured research data to a text file."
)


# --- Search tool using API wrapper directly (bug-free) ---
class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up online")

search_api = DuckDuckGoSearchAPIWrapper()

def search_tool_func(query: str):
    """Search the web for information."""
    return search_api.run(query)

search_tool = StructuredTool.from_function(
    func=search_tool_func,
    name="search",
    description="Search the web for information about a topic.",
    args_schema=SearchInput
)


# --- Wikipedia tool ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
