from typing import TypedDict ,List ,Optional 
from langchain_core.messages import BaseMessage

class AgentState (TypedDict):
    messages: List[BaseMessage]
    intent : str 
    lead_name : Optional[str]
    lead_email : Optional[str]
    lead_platfrom : Optional[str]
    lead_captured : bool 