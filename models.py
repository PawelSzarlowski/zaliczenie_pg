from pydantic import BaseModel

class Userquery(BaseModel):
    job_desc: str

class SearchParams(BaseModel):
    location: str
    category: str
    subCategory: str
    experience: str
    minimumSalary: int