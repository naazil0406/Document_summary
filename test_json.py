from services.json import JSONParser

parser = JSONParser()
pages = parser.extract_pages("sample.json")
for p in pages:
    print(p.text)