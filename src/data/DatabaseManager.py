import os
import time

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId, json_util


class DatabaseManager:
    
    def __init__(self):
        conn_str = os.environ.get("MONGO_CONN_STR", "mongodb://172.26.0.1:27017")
        self.client = MongoClient(conn_str)
        self.db: Database = self.client.docparser
        self.coll: Collection = self.db.docs
    
    
    def get(self, id: str) -> any:
        return self.coll.find_one({"_id": ObjectId(id)})
    
    
    def getAll(self) -> (list|None):
        return list(self.coll.find())
    
    
    def saveToTemp(self, docs: list, tempDir: str = "./tmp") -> str:
        tmp_dir = os.path.join(tempDir, str(round(time.time())))
        os.makedirs(tmp_dir)
        
        for doc in docs:
            doc_id = str(doc["_id"])
            tmp_file_path = os.path.join(tmp_dir, f"{doc_id}.json")
            
            with open(tmp_file_path, "w") as tmp_file:
                # combined_text = "\n\n".join([page["Text"] for page in doc["Data"]["Pages"]])
                # tmp_file.write(combined_text)
                tmp_file.write(json_util.dumps(doc["Data"]))
        
        return tmp_dir