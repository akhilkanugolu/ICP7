# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:07:45 2020

@author: akhil
"""

from bs4 import BeautifulSoup
import urllib.request


def search_spider():

    url = "https://en.wikipedia.org/wiki/Google"
    source_code = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")

    body = soup.find('div', {'class': 'mw-parser-output'})
    file.write(str(body.text))

