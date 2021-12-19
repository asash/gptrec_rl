import mmh3

from aprec.api.user import User


class HashingUserFeaturizer(object):
    def __init__(self, num_cat_hashes=3, cat_hashes_space=1000):
        self.num_cat_hashes = num_cat_hashes
        self.cat_hashes_space = cat_hashes_space

    def __call__(self, user: User):
        result = []
        for feature in user.cat_features:
            for hash_num in range(self.num_cat_hashes):
                val = f"{feature}_" + str(user.cat_features[feature]) + f"_hash{hash_num}"
                hash_val = mmh3.hash(val) % self.cat_hashes_space + 1
                result.append(hash_val)
        return result
