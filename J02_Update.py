##########################################################################
# Update Joker
from bs4 import BeautifulSoup
import urllib.request
import os  # accessing directory structure
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os
import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')


url = "https://www.loto49.ro/arhiva-joker.php"
hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}

req = urllib.request.Request(url, headers=hdr)
response = urllib.request.urlopen(req)
resp = response.read()

soup = BeautifulSoup(resp, 'html.parser')

data = []
table = soup.find("table")
for row in table.findAll("tr"):
    cells = row.findAll("td")
    if len(cells) == 7:
        date = cells[0].find(text=True)
        n1 = cells[1].find(text=True)
        n2 = cells[2].find(text=True)
        n3 = cells[3].find(text=True)
        n4 = cells[4].find(text=True)
        n5 = cells[5].find(text=True)
        j = cells[6].find(text=True)
        data.append([str(date), n1, n2, n3, n4, n5, j])

df = pd.DataFrame(data, columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'j'])
df = df.drop(0)

df = df[pd.to_datetime(df['date']) >= '2014-01-01']
df['date'] = pd.to_datetime(df['date'])
df['weekd'] = pd.to_datetime(df['date']).dt.day_name()
df[['n1', 'n2', 'n3', 'n4', 'n5']] = df[['n1', 'n2', 'n3', 'n4', 'n5']].apply(
    pd.to_numeric, errors='coerce')
df[['j']] = df['j'].str[2:].apply(pd.to_numeric, errors='coerce')
df["avg"] = (df['n1']+df['n2']+df['n3']+df['n4']+df['n5'])/5
df['idx'] = range(1, len(df) + 1)

dir_path = os.path.abspath('')
df.to_csv(dir_path + '/jokerFULL.csv', index=False, header=True)
dfJ = df[['date', 'j']]
dfJ.to_csv(dir_path + '/joker.csv', index=False, header=True)
