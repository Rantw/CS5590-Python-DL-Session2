# Ronnie Antwiler
# 3 Consider the following scenario. You have a list of students who are attending class "Python" and another list
# of students who are attending class "Web Application".
#
# Find the list of students who are attending both the classes.
# Also find the list of students who are not common in both the classes.
#
# Print the both lists. Consider accepting the input from the console for list of students that belong to
# class “Python” and class “Web Application”.

webClass = []
pythonClass = []
bothClass = []
oneClass = []
cont = 'Y'

while cont != 'n' and cont != 'N':
    string = input("Please enter the student name then which classes they are in: "
                   "(separate name and classes with a comma and space): ")
    info = tuple(string.split(', '))

    if info[1][0] == 'w' or info[1][0] == "W":
        webClass.append(info[0])
    elif info[1][0] == 'p' or info[1][0] == "P":
        pythonClass.append(info[0])
    if len(info) == 3:
        if info[2][0] == 'w' or info[2][0] == "W":
            webClass.append(info[0])
        elif info[2][0] == 'p' or info[2][0] == "P":
            pythonClass.append(info[0])
    cont = input("Add another student (Y/N): ")

# Sort through the students listed in the python class and check if they are also in the web class. If they are add
# them to the both classes list. If they are not, then add them to the only one class list.
for student in pythonClass:
    if student in webClass:
        bothClass.append(student)
    else:
        oneClass.append(student)

# Sort through the web class list and see if they are in the python class. If they are in the python class, then they
# would have already been added to the both classes list. Just those students only in the web class need to be added
# to one class list
for student in webClass:
    if student not in pythonClass:
        oneClass.append(student)

print("Students in the Python class are:", ', '.join(sorted(pythonClass)))
print("Students in the Web Development class are:", ', '.join(sorted(webClass)))
print("Students in both classes are:", ', '.join(sorted(bothClass)))
print("Students in only one class are:", ', '.join(sorted(oneClass)))