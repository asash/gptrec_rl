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

    def __str__(self):
        return "item id={} title={} tags={}".format(self.item_id, self.title, self.tags)

    def __repr__(self):
        return self.__str__()
