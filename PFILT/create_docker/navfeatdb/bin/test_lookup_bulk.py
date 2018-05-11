from pymongo import MongoClient
import cProfile

db = MongoClient().tilefeatures

def test():
    size = db.rows.find({ "$and": [{"lat": {"$gte": 41.023}},{"lat": {"$lte": 41.024}},{"lon": {"$gte": -80.889}},{"lon": {"$lte": -80.888}}] }).count()
    print(size)

cProfile.run("test()")

#lon: -80.88874825995401
#lat: 41.023517593802666