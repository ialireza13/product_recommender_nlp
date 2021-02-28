import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import sys

from config import urls, page_url_id, page_404_title, product_class_id, product_title_id

def try_to_parse(x):
    try:
        return x.find('a', class_=product_class_id).text
    except:
        return '\n'

# number of pages to crawl for each starting url
max_page = int(sys.argv[1])+1

products = 0
with open("products/products.txt", "w") as f:
    for raw_url in tqdm(urls):
        page_no = 1
        while page_no<max_page:
            url = raw_url + page_url_id + str(page_no)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            title = soup.title.string
            if title == page_404_title: break
            title = title.split('|')[0]
            results = soup.find_all('div', class_=product_title_id)
            parsed = [title + try_to_parse(res) for res in results]
            parsed = [x for x in parsed if '\n' not in x]
            for item in parsed:
                f.write(item+'\n')
            products += len(parsed)
            page_no += 1