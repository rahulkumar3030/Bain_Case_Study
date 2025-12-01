from pydantic import BaseModel
from typing import List, Dict, Optional

# ------ Models ------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    bot_message: str

class EndSessionRequest(BaseModel):
    session_id: str    