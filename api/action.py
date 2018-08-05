import json
class Action(object):
    def __init__(self, user_id, item_id, timestamp, data={}):
        self.user_id = user_id
        self.item_id = item_id
        self.data = data
        self.timestamp = timestamp

    def to_str(self):
       result = "Action(uid={}, item={}, ts={}".format(
                    self.user_id, 
                    self.item_id, 
                    self.timestamp)
       if self.data != {}:
           result += ", data={}".format(json.dumps(self.data))
       result += ")"
       return result

    def __str__(self):
        return self.to_str()
        
    def __repr__(self):
        return self.to_str()
        
