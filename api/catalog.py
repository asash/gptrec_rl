class Catalog(object):
    def __init__(self):
        self.items = {}

    def add_item(self, item):
        self.items[item.item_id] = item

    def get_item(self, item_id):
        return self.items[item_id]
