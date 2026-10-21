from app.services.search import hybrid_search


results = hybrid_search("best phone for low light photography", k=3)

for document, score in results:
    print(score, document)
