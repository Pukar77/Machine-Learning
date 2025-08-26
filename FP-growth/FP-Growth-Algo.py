import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Dataset
dataset = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Milk", "Bread", "Butter", "Jam"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter"],
]

# One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


print(df)

# FP-Growth with min support = 50% (0.5)
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

print("\nFrequent Itemsets (support >= 50%):\n")
print(frequent_itemsets)
