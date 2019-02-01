# Ronnie Antwiler
# 3. Write a python program to swap cases. In other words, convert all lowercase letters to
# uppercase letters and vice versa.
# For
# Example:
# Www.HackerRank.com
# →
# wWW.hACKERrANK.COM

# Input String from user
string1 = input("Please enter your string: ")

# Convert string to a list
string2 = list(string1)

# Set up for loop to modify list with capitalization changes
i=0
for i in range(len(string1)):
    if string1[i].isupper():
        string2[i]=string1[i].lower()
        i += 1
    elif string1[i].islower():
        string2[i]=string1[i].upper()
        i += 1

# Join the list values together into string for output
output="".join(string2)

print("Original String:", string1)
print("Modified String:", output)