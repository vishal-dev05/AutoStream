from langchain.tools import tool 

@tool
def mock_lead_capture(name : str , emial : str ,platform : str ) -> str:
    """ Captures a lead after collecting name, email, and creator platform.
    Only call this when you have all three values confirmed by the user.
    """
    
    print(f"Captured lead: Name: {name}, Email: {emial}, Platform: {platform}")
    return(f"Lead captured successfully for {name}")
