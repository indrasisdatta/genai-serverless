from typing import Optional

from pydantic import BaseModel

class UserQuery(BaseModel):
    user_question: str
    session_id: Optional[str]