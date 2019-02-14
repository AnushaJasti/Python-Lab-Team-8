import requests #importing the request library
from bs4 import BeautifulSoup
infile=open("out.html",'w')
html = requests.get("https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States") #get requests from the mentioned url
soup = BeautifulSoup(html.content, "html.parser") #calling html parser
req_list=soup.find('table',{"class" : "wikitable sortable plainrowheaders"})
infile.write(str(req_list))


