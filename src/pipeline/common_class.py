from pydantic import BaseModel


class QueryBody(BaseModel):
    beginTime: str
    endTime: str