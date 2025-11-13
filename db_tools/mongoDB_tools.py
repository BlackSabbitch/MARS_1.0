from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# from pymongo.mongo_client import MongoClient


class MongoTools:
    def __init__(self, username, password, cluster_name, app_name):
        uri = f"mongodb+srv://{username}:{password}@{cluster_name}.mongodb.net/?appName={app_name}"
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def get_database(self, db_name):
        return self.client[db_name]