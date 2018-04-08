class Item(object):
    tags = None
    title = None

    def __init__(self, item_id):
        self.item_id = item_id

    def with_tags(self, tags):
        self.tags = tags
        return self

    def with_title(self, title):
        self.title = title
        return self
