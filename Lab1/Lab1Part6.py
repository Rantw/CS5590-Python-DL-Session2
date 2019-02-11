# Ronnie Antwiler
# 6. Program a code which download a webpage contains a table using Request library, then parse the page
# using Beautiful soup library. You should save the information about the states and their capitals in a file.
# Sample input:
# https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States
# Sample output:
# Save the table in this link into a file

import requests
from bs4 import BeautifulSoup
import os

url = 'https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States'

'''
if not os.path.exists('attempt'):
    print("Creating folder ")
    file2 = open('attempt.txt', 'a+', encoding='utf-8')

source = requests.get(url)
text = source.text
soup = BeautifulSoup(text, "html.parser")
tables = soup.findAll('table')
table = tables[0]
file2.write(str(table.text))
print(table.text)
'''


source = requests.get(url)
text = source.text
soup = BeautifulSoup(text, "html.parser")
tables = soup.findAll('table')
table = tables[0]

table_rows = table.findAll('tr')

tr=3

table_columns = table_rows[tr].findAll('td')
print(table_columns[0].text)
print(table_columns[1].text)
table_columns = table_rows[tr+1].findAll('td')
print(table_columns[0].text)
print(table_columns[1].text)
table_columns = table_rows[tr+2].findAll('td')
print(table_columns[0].text)
print(table_columns[1].text)
