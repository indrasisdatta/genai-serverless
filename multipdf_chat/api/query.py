from fastapi import HTTPException
from ..helper import user_input
from dotenv import load_dotenv
import os
load_dotenv()

def query_answer(user_question, session_id, request):
    try:
        return {
            "data": user_input(user_question, session_id, request)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))