"""
A web scrpaing program for gathering headlines frm WSJ
"""
import requests
import bs4

def main():

    headlines = gather_headlines(2023, 2, 14)
    print_results(headlines)
    
def gather_headlines(year, month, day):

    dictionary = {}

    url = 'https://www.wsj.com/news/archive/' + str(year) + "/" + str(month) + "/" + str(day)
    # User-Agent specification from Andrej Kesely (https://stackoverflow.com/users/10035985/andrej-kesely)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}

    request = requests.get(url, headers=headers).content
    soup = bs4.BeautifulSoup(request, 'html.parser')

    for article in soup.select('article'):
        header = article.span.text
        text = article.h2.text

        if header in dictionary.keys(): 
            dictionary[header].append(text)
                    
        else: 
            dictionary[header] = [text]

    return dictionary

def print_results(quotes):

    print("\n")

    for key, value in quotes.items():
        print(key + ":", "\n")

        for item in range(0, len(value)):
            print("–", value[item])

        print("\n")

main()