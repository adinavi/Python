import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["lib"]
books = db["books"]
# print(
#     books.update_one({
#         "author": "jim collins"
#     }, {
#         "$set": {
#             "published": 2021
#         }
#     }).modified_count
# )
books.update_many({"published" : 2021}, {
    "$set": {
        "name": "PUBLISHED IN 2021"
    }
})
print([*books.find()])
