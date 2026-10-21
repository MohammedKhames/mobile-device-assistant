
from pydantic import BaseModel,Field
from typing import Literal


# Device Verification
class Device(BaseModel):
    brand: str
    model: str
    release_year: int
    specs: dict[ str , str ]
    price_tier: Literal[ "budget" , "mid-range" , "flagship"]

    
    # Literal vs Enum 
       
       # Literal ==> its simple for small Group
             # Literal["a" ,"b","c"]
       
       
       # Enum ==> if you will use it in many Places in the code
       
            # class PriceTier(str, Enum):
                # BUDGET = "budget"
                # MID_RANGE = "mid-range"
                # FLAGSHIP = "flagship"

                # price_tier: PriceTier
    
# ----------------------------------------------------- #
# FastApi
    
    
### Extract    
# Input sent to POST 
# Input sent to POST /extract.
class ExtractRequest(BaseModel):
    text: str



# Output returned by POST 
class ExtractResponse(Device):
    pass





#### Search
# Input sent to POST 
class SearchRequest(BaseModel):
    q: str = Field(max_length=500)


# One document in the search results
class SearchResult(BaseModel):
    document: str
    score: float


# Output returned by POST
class SearchResponse(BaseModel):
    results: list[SearchResult]
    
    
    
    
    
    
#---------------------------------------
# input validation

