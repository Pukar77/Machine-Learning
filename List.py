#It’s an ordered, mutable collection that can store multiple items (of any data type) in a single variable.

fruits = ["apple", "banana", "mango"]

print(fruits)
print(fruits[0])

for fruit in fruits:
    print(fruit)
    
for index, fruit in enumerate(fruits):
    print(f"{index}, ", f"{fruit}")