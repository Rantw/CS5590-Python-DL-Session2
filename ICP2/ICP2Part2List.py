# Ronnie Antwiler
# 2. Write a python program to implement Stack and Queue data structures using “List” and
# its functions in python:
# Refer this link:
# https://docs.python.org/3/tutorial/datastructures.html

# Stack Operations
# Get Stack
string1 = input("Please enter the stack:")
stack = string1.split(" ")

num1 = int(input("Input 0 to push, 1 to pop, 2 to end session"))
while num1 != 2:
    if num1 == 0:
        stack.append(input("Please enter what to push:"))
    elif num1 == 1:
        stack.pop()
    else:
        num1 =2
    num1 = int(input("Input 0 to push, 1 to pop, 2 to end session:"))

print(stack)

# Queue Operations
# Get Queue
string3 = input("Please enter the queue:")
lst = string3.split(" ")

#initialize queue
from collections import deque
queue = deque(lst)

num1 = int(input("Input 0 to add to queue, 1 to remove from queue, 2 to end session"))
while num1 != 2:
    if num1 == 0:
        queue.append(input("Please add to the queue:"))
    elif num1 == 1:
        queue.popleft()
    else:
        num1 =2
    num1 = int(input("Input 0 to add to queue, 1 to remove from queue, 2 to end session:"))

print(queue)