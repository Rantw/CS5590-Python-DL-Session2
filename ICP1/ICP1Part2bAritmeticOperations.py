# Ronnie Antwiler
# 2b. Take two numbers from user and perform arithmetic operations on them

# Get input from users
num1String = input('Please enter a number: ')
num2String = input('Please enter a second number: ')

# Convert input to float
num1 = float(num1String)
num2 = float(num2String)

# Get input again if zero was second number
if num2 == 0.0:
    num2String = input('Please enter a number besides zero:')
    num2 = float(num2String)

# Perform Arithmetic Operations
addnum = num1+num2
subnum = num1-num2
multnum = num1*num2
divnum = num1 / num2

# Display results and limiting decimal points to 2s
print('The numbers added together are', "%.2f" % addnum)
print('The second number subtracted from the first is', "%.2f" % subnum)
print('The numbers multiplied together are', "%.2f" % multnum)
print('The first number divided by the second number', "%.2f" % divnum)

