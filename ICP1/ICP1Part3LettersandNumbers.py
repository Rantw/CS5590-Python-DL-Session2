# Ronnie Antwiler
# 3. Write a program that accepts a sentence and prints the number of letters and digits in
# Sentence.

# Get Sentence from user
string = input('Please enter a sentence:')

# Set Initial values to zero
num = 0
let = 0

# Count digits and alphabet letters
for i in range(len(string)):
    if string[i].isdigit():
        num +=1
    elif string[i].isalpha():
        let +=1

# Display results
print('letters:', let, '  ', 'numbers:', num)