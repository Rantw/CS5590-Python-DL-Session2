# Ronnie Antwiler
# 5. Write a python program to create any one of the following management systems.
# b.
# Airline Booking Reservation System (e.g. classes Flight, Person, Employee, Passenger etc.)
# Prerequisites:
# a. Your code should have at least five classes
# b. Your code should have _init_ constructor in all the classes
# c. Your code should show inheritance at least once
# d. Your code should have one super call
# e. Use of self is required
# f. Use at least one private data member in your code
# g. Use multiple Inheritance at least once
# h. Create instances of all classes and show the relationship between them
# Comment your code appropriately to point out where all these things are present

class employee:
    employeeCount = 0

    def __init__(self, n1, n2, n):
        self.first_name = n1
        self.last_name = n2
        self.__ID_number = n
        employee.employeeCount += 1

    def resetEmployeeCount():
        employee.employeeCount = 0

class passenger:
    passCount = 0

    def __init__(self, n1, n2):
        self.first_name = n1
        self.last_name = n2
        passenger.passCount += 1

    def resetpassengers():
        passenger.passCount = 0

class person(employee, passenger):
    def __init__(self, t, n1, n2, n):
        self.type = t
        if self.type == 'E' or self.type == 'e':
            employee.__init__(self, n1, n2, n)
        elif self.type == 'P' or self.type == 'p':
            passenger.__init__(self, n1, n2)

    def printperson(self):
        if self.type == 'E' or self.type == 'e':
            print("Employee Name:", self.first_name, self.last_name)
        elif self.type == 'P' or self.type == 'p':
            print("Passenger Name:", self.first_name, self.last_name)


class flight:
    def __init__(self, n, t, s):
        self.flight_number = n
        self.seat_type = t
        self.seat = s

    def printflight(self):
        print('Flight Number:', self.flight_number, 'Seat:', self.seat_type, self.seat)

class booking(person,flight):
    def __init__(self, t, n1, n2, n, num, st, sn):
        super().__init__(t, n1, n2, n)
        flight.__init__(self, num, st, sn)

    def printBookings(self):
        person.printperson(self)
        flight.printflight(self)

book = []
cont = 'Y'

while cont != 'n' and cont!= 'N':
    type = input("Employee or Passenger: (E/P) ")
    string = input("Please enter the employee/passenger info: (First Name, Last Name, Employee ID (0 if passenger): "
                   "(No Commas. Just Spaces)")
    string1 = input("Please enter the flight information: Flight Number, Seat Type/Job, and Seat Number (0 if employee)"
                    "(No Commas. Just Spaces) ")
    info1 = string.split(' ')
    info2 = string1.split(' ')

    if type == 'E' or type == 'e':
        info2[2] = int(0)
    if type == 'P' or type == 'p':
        info1[2] = int(0)

    info1[2] = int(info1[2])
    info2[0] = int(info2[0])
    info2[2] = int(info2[2])

    book.append(booking(type, info1[0], info1[1], info1[2], info2[0], info2[1], info2[2]))
    cont = input("Enter additional Flight information (Y/N): ")

i = 0
for i in range(len(book)):
    book[i].printBookings()

print("The number of employees is", employee.employeeCount)
print("The number of passengers is", passenger.passCount)