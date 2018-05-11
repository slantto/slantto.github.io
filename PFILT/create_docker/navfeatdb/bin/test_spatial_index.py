from pymongo import MongoClient, GEO2D
import cProfile
import matplotlib.pyplot as plt
import numpy as np

db = MongoClient().tilefeatures
point = { "type": "Point", "coordinates": [41.0235, -80.8885] }
query = { "loc": { "$nearSphere": { "$geometry": { "type": "Point", "coordinates": [-80.8885, 41.0235] }, "$maxDistance": 100 } } }
result = None
def test():
    global result
    result = db.rows.find(query)


cProfile.run("test()")

print(result.count())
arr = np.array([(each["loc"]["coordinates"][0], each["loc"]["coordinates"][1]) for each in result])
plt.plot(arr[:, 0], arr[:, 1], 'x')
plt.show()