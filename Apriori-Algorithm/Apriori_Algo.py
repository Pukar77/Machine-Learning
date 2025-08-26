import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder



# Step 1: Sample transactions
dataset = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Milk", "Bread", "Butter", "Jam"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter"],
]

# Step 2: Convert to one-hot encoded dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


print(df)

# Step 3: Apply Apriori with minimum support = 0.5 (50%)
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("\nFrequent Itemsets (support >= 50%):\n")
print(frequent_itemsets)
