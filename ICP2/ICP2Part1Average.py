# Ronnie Antwiler
# 1. Write a python program to take the total number of plants as an input on first line and all
# the space separated heights of the plants on a second line. Finally Output the average
# height value on a single line to three decimals.
# Input:
# 10
# 161 182 161 154 176 170 167 171 170 174
# Output:
# 169.375

# Input values from the user
num1 = int(input("Please enter the number of heights:"))
string1 = input("Please enter the values of each height separated by a space:")

# Separate the heights apart from the string
lst = string1.split(" ")

# Make sure the correct number of heights are entered
while len(lst) != num1:
    string1 = input("Number missing. Please enter the values of each height separated by a space:")
    lst = string1.split(" ")

# Convert List of values from a string to floating values
i=0
for i in range(num1):
    lst[i] = float(lst[i])

# Perform calculations for average
ave1 = sum(lst)/ num1

print("The average of the heights is", ave1)