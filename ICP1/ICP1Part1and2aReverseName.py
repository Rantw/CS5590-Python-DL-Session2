# Ronnie Antwiler
# 1. State differences between Python 2 and Python 3 version

# Print is treated differently between the two versions. Print is a statement in version 2
# while it is a function in version 3. This affects the way that information to be printed is passed.

# Version 2 treats integers differently as version 3. In version 2, numbers entered without digits after
# a decimal point are considered integers. This led to problems when doing division. 5/2 would be 2 in version 2.
# To get 2.5, 5.0/2.0 needs to be typed plus instruct the answer to be returned as float.
# For version 3, 5/2 is 2.5 default.

# In previous version of Python, variables that are iterated over in a list comprehension and
# global variables that share the same name can cause the global variable to be changed which is not something
# that is desired. Python 3 fixes this.

# Version 3 stores strings as unicode by default. Version 2 did not do this.

# Version 2 and version 3 also have some different syntax for different things like raising errors.


# 2a. Take the user first name and last name and print it in reversed form

#Get names from user
firstName = input('Please enter your first name: ')
lastName = input('Please enter your last name: ')

#Reverse first and last names
string1 = firstName[::-1]
string2 = lastName[::-1]

#Print reversed names
print(string1, string2)

