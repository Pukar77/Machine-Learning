import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Step 1: Create dataset
data = pd.DataFrame([
    ['High', 'Good', 'Yes'],
    ['High', 'Poor', 'Yes'],
    ['Low', 'Good', 'Yes'],
    ['Low', 'Poor', 'No'],
    ['High', 'Good', 'Yes'],
    ['Low', 'Poor', 'No'],
    ['Low', 'Good', 'Yes'],
    ['High', 'Poor', 'Yes'],
    ['Low', 'Poor', 'No'],
    ['High', 'Good', 'Yes']
], columns=['Hours_Studied', 'Attendance', 'Pass'])

# Step 2: Define Bayesian Network structure
model = BayesianModel([('Hours_Studied', 'Pass'), ('Attendance', 'Pass')])

# Step 3: Learn CPDs (Conditional Probability Distributions)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Step 4: Perform inference
infer = VariableElimination(model)

# Step 5: Predict probability of passing if Hours_Studied='Low' and Attendance='Poor'
q = infer.query(variables=['Pass'], evidence={'Hours_Studied': 'Low', 'Attendance': 'Poor'})
print(q)
