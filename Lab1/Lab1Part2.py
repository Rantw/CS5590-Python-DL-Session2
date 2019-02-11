# Ronnie Antwiler
# 2. Suppose you have a list of tuples as follows:
# [( ‘John’, (‘Physics’, 80)) , (‘ Daniel’, (‘Science’, 90)),
# (‘John’, (‘Science’, 95)), (‘Mark’,(‘Maths’, 100)), (‘Daniel’, (’History’, 75)),
# (‘Mark’, (‘Social’, 95))]
# Create a dictionary with
# keys as names and values as list of (subjects, marks) in sorted order.
# {
# John : [(‘Physics’, 80), (‘Science’, 95)]
# Daniel : [ (’History’, 75), (‘Science’, 90)]
# Mark : [ (‘Maths’, 100), (‘Social’, 95)]
# }

# Create list of Names and scores.
lst = tuple([('John', ('Physics', 80)), ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark', ('Maths', 100)),
       ('Daniel', ('History', 75)), ('Mark', ('Social', 95))])


# Create empty Dictionary
scores = {}

for grade in lst:
    # Check to see if the Student is in the Dictionary already. Student Name is the key for the dictionary
    # If Student is in dictionary then associate the grade to his or her name (Key)
    # Else Add Student name to the dictionary with a grade
    if grade[0] in scores:
        scores[grade[0]] = scores[grade[0]] + grade[1]
    else:
        scores[grade[0]] = grade[1]

print(sorted(scores.items()))
print(scores['Daniel'])
print(scores['John'])
print(scores['Mark'])