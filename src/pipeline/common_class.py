from pydantic import BaseModel


class QueryBody(BaseModel):
    day: str
    beginTime: str
    endTime: str