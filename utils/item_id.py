class ItemId(object):
    def __init__(self):
        self.straight = {}
        self.reverse = {}

    def size(self):
        return len(self.straight)

    def get_id(self, item_id):
        if item_id not in self.straight:
            self.straight[item_id] = len(self.straight)
            self.reverse[self.straight[item_id]] = item_id
        return self.straight[item_id]

    def has_id(self, id):
        return id in self.reverse

    def has_item(self, item_id):
        return item_id in self.straight

    def reverse_id(self, id):
        return self.reverse[id] 


