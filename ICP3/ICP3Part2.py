# Ronnie Antwiler
# 2. Web scraping
# Write a simple program that parse a Wiki page mentioned below and follow the instructions:
# https://en.wikipedia.org/wiki/Deep_learning
# 1. Parse the source code using the Beautiful Soup library and save the parsed code in a variable
# 2. Print out the title of the page
# 3. Find all the links in the page (‘a’ tag)
# 4. Iterate over each tag(above) then return the link using attribute "href" using get

import requests
from bs4 import BeautifulSoup
import os

url = 'https://en.wikipedia.org/wiki/Deep_learning'
source = requests.get(url)
plain = source.text
soup = BeautifulSoup(plain, 'html.parser')
for div in soup.findAll('a'):
    print(div.get('href'))
