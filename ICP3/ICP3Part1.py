# Ronnie Antwiler
# 1. Create a class Employee and then do the following
# a. Create a data member to count the number of Employees
# b. Create a constructor to initialize name, family, salary, department
# c. Create a function to average salary
# d. Create a Fulltime Employee class and it should inherit the properties of Employee class
# e. Create the instances of Fulltime Employee class and Employee class and call their member functions.

class Employee:
    count = 0
    totalSalary = 0

    def __init__(self,n,f,s,d):
        self.name = n
        self.family = f
        self.salary = s
        self.department = d
        Employee.count += 1
        Employee.totalSalary += s

    def printEmployee(self):
        print("Employee: ", self.name, ' ', self.family)

    def averSalary():
        return Employee.totalSalary / Employee.count

class fulltime(Employee):
    def __init__(self, n, f, s, d):
        Employee.__init__(self, n, f, s, d)


e = []
cont = 'Y'

while cont != 'n' and cont!= 'N':
    string = input("Please enter the employee information: (No Commas. Just Spaces)")
    emt = input("Full time Employee: (Y/N) ")
    if emt == 'n' or emt == 'N':
        info = string.split(' ')
        info[2] = float(info[2])
        e.append(Employee(info[0],info[1],info[2],info[3]))
    elif emt == 'y' or emt == 'Y':
        info = string.split(' ')
        info[2] = float(info[2])
        e.append(fulltime(info[0],info[1],info[2],info[3]))
    cont = input("Enter another employee (Y/N): ")

print('The average salary is: ', Employee.averSalary())

i = 0
for i in range(len(e)):
    e[i].printEmployee()

