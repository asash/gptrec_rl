import json
class Action(object):
    def __init__(self, user_id, item_id, timestamp, data={}):
        self.user_id = user_id
        self.item_id = item_id
        self.data = data
        self.timestamp = timestamp

    def to_str(self):
        return "user_id={}, item_id={}, timestamp={}, data={}".format(
                    self.user_id, 
                    self.item_id, 
                    self.timestamp, 
                    json.dumps(self.data))
        
