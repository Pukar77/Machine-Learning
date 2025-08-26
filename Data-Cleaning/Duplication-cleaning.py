import pandas as pd 
# Step 1: Create dataset  
data = { 
    "Name": ["Pukar", "Rimal", "Ram", "Hari", "Krishna", "Ronaldo",          
"Neymar", "Neymar"], 
 "Excersie_Name": ["Push Ups", "Running", "Cycling", "Yoga", "Swimming", "Squats", "Skipping", "Skipping"], 
    "Duration": [15, 130, 45, 60, 150, 25, 12, 12],       # Duration in minutes
    "CaloriesBurned": [100, 300, 450, 200, 400, 250, 70, 70],  # Calories burned
    "Intensity": ["Medium", "High", "High", "Low", "High", "Medium", "Low", "Low"], 
} 
df = pd.DataFrame(data) 
print("Original Dataset:\n") 
print(df) 
# Step 2: Delete rows where Duration > 120 
df_filtered = df[df["Duration"] <= 120] 
print("\nDataset after removing Duration > 120:\n") 
print(df_filtered) 
# Step 3: Find duplicate rows 
duplicates = df_filtered[df_filtered.duplicated()] 
print("\nDuplicate Rows:\n") 
print(duplicates) 
# Step 4: Count duplicates 
print("\nNumber of duplicate rows:", len(duplicates))