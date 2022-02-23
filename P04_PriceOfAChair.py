#02 Price of a chair

import requests
from bs4 import BeautifulSoup


r = requests.get("http://www.johnlewis.com/store/john-lewis-wade-office-chair-black/p447855")
soup = BeautifulSoup(r.content, "html.parser")
element = soup.find("h2", {"class": "attention-header"})
string_value = element.text.strip() #"£115.00"

ttext = string_value[0:]

print(ttext)
