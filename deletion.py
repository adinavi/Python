import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["lib"]
books = db["books"]

books.delete_many({
    "published": 2021,
})

print([*books.find()])
