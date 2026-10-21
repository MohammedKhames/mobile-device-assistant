from fastapi import FastAPI, HTTPException

from app.schema.device import ( Device, ExtractRequest, ExtractResponse, SearchRequest, SearchResponse, SearchResult,)
from app.services.extraction import extract_device
from app.services.search import hybrid_search
from app.utils.security import PromptInjectionError


app = FastAPI(title="Mobile Device Assistant API")

#### Endpoints 

# Extract 
@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> Device:
    # Extract and validate device data from the submitted text
    try:
        return extract_device(request.text)
    except PromptInjectionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

# Search
@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    # Convert the search service's tuples into API result models
    matches = hybrid_search(request.q)

    results = [
        SearchResult(document=document, score=score)
        for document, score in matches
    ]

    return SearchResponse(results=results)