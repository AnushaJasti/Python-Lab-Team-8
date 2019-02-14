import requests
from bs4 import BeautifulSoup
infile=open("out.html",'w')
html = requests.get("https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States")
soup = BeautifulSoup(html.content, "html.parser")
req_list=soup.find('table',{"class" : "wikitable sortable plainrowheaders"})
infile.write(str(req_list))
