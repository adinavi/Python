import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["lib"]
books = db["books"]
query = {
    "name": "Life 3.0"
}
# print([*books.find({
#     "name": "Life 3.0"
# }, {
#     "author": 1
# })])
print([
    *books.find().sort("published", 1).limit(1)
])
