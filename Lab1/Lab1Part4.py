# Ronnie Antwiler
# 4. Given a string, find the longest substring without repeating characters along with the length.
# Input: "pwwkew"Output: wke,3

# Get string from user
stringinput = input('Please enter a string:')

# Two lists are needed. One will be a list of characters that make up the current string that is being analyzed
# The other list is a list of all strings found while running the program.
current = []
strings = []

# Look at each character in the Input String
for char in stringinput:
    # Check to see if character is in current string list
    if char in current:
        # Use the join function on the current to create a string with nothing in between the characters. Then use
        # append to add the new string to the list of strings found
        strings.append(''.join(current))
        # Add 1 to the current position of the char in the current list. Then set the current list equal to the
        # current list from index spot + 1 to the end. This will reset the current list to blank because there is
        # no values there. If the plus one is not added then the next current list would not be reset to blank but
        # instead carry over the current char.
        nextstring = current.index(char)+1
        current = current[nextstring:]
    # Add current character to current string list after checking to see if it is already in the list.
    current.append(char)

# Add the last string being analyzed to the list of strings found
strings.append(''.join(current))

# Use the max function to grab the longest string from the strings list. A key has to be sent to the function.
# The key tells the function by which criteria to pick the longest string. In this case, the length of the found
# strings are what is desired. If key was not used then the function would not return the correct string all the time.
long = max(strings, key = len)

print('The longest string of characters without repeating is', long,'with a length of', len(long))