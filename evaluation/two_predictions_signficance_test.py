import sys
import gzip
import json
import pandas as pd
from scipy.stats import ttest_ind

predictions_file_1 = sys.argv[1]
predictions_file_2 = sys.argv[2]


def read_data(filename):
    result = []
    with gzip.open(filename) as input:
        for line in input:
            result.append(json.loads(line)['metrics'])
    return pd.DataFrame(result)

df1 = read_data(predictions_file_1)
df2 = read_data(predictions_file_2)

overlap_columns = set(df1.columns).intersection(set(df2.columns))


docs = []

for column_name in overlap_columns:
    df1_series = df1[column_name]
    df2_series = df2[column_name]

    mean1 = df1_series.mean()
    mean2 = df2_series.mean()
    doc = {}
    doc["metric_name"] = column_name
    doc["mean1"] = mean1
    doc["mean2"] = mean2
    doc["difference"] = mean2 - mean1
    doc["difference_pct"] = (mean2 - mean1) * 100 / mean1
    t, pval = ttest_ind(df1_series, df2_series) 
    doc["p_value"] = pval 
    docs.append(doc)

print(pd.DataFrame(docs))

