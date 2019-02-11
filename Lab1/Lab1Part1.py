# Ronnie Antwiler
# 1. Write a program that computes the net amount of a bank account based a transaction log from
# console input. The transaction log format is shown as following:
# Suppose the following input is supplied to the program:
# Deposit 300
# Deposit 250
# Withdraw 100
# Deposit 50
# Then the output should be
# Total amount - $500


amount = 0.00
end = 0

while end != 1:
    trans = input("Please enter a transaction (Put a space between the transaction type and the amount) "
                  "or hit any key besides d or w to compute total: ")
    lst = trans.split(" ")
    letter = lst[0][0]
    if letter == 'd' or letter == 'D':
        amount = amount + float(lst[1])
    elif letter == 'w' or letter == 'W':
        amount = amount - float(lst[1])
    else:
        end = 1

print("The total amount - ", amount)