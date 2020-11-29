import sys
import json
from copy import deepcopy
import pandas as pd

data = json.load(open(sys.argv[1]))
i = 0
for split_fraction in data:
    print("="*40)
    i += 1
    doc = deepcopy(split_fraction)
    recommenders = doc['recommenders']
    del(doc['recommenders'])
    print("experiment_{}".format(i))
    print (pd.DataFrame([doc]).T)
    print("\n")

    experiment_docs = []
    for recommender_name in recommenders:
        recommender = recommenders[recommender_name]
        recommender['name'] = recommender_name
        del(recommender['model_metadata'])
        experiment_docs.append(recommender)

    df = pd.DataFrame(experiment_docs)
    df = df.sort_values("ndcg@40")
    df = df.set_index('name')
    print(df)
        


