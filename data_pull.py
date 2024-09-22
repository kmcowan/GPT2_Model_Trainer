import requests
from bs4 import BeautifulSoup

import requests
import docx

import text_cleaner

global cursor_mark
import xml.etree.ElementTree as ET

def fetch_from_solr(solr_url, query="*:*", rows=1000):
    all_text = []
    global cursor_mark
    cursor_mark = "*"
    params = {
        "q": query,
        "rows": rows,
        "wt": "json",
        "sort": "id asc",
        "cursorMark": cursor_mark
    }
    curr_count = 0

    while curr_count < rows:
        response = requests.post(f"{solr_url}/select", data=params)
        response.raise_for_status()
        data = response.json()

        # Concatenate the text fields from all the documents
        all_text.extend([doc.get('text', '') for doc in data['response']['docs']])
        curr_count += len(data['response']['docs'])

        # Check if there are more results
        next_cursor_mark = data.get('nextCursorMark')
        if next_cursor_mark == cursor_mark:
            break
        cursor_mark = next_cursor_mark
        params["cursorMark"] = cursor_mark

    return " ".join(all_text)


# Example usage:
solr_url = 'http://localhost:8983/solr/training_data'
query = '*:*'  # Default to all documents
solr_text = fetch_from_solr(solr_url, query)


def fetch_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = text_cleaner.clean_text(text)
    return text


# Example usage:
# file_path = 'data/my_text_file.txt'
# file_text = fetch_from_file(file_path)


def fetch_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract the main text from the page (you may need to customize this depending on the webpage structure)
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        raise ValueError(f"Failed to fetch URL. Status code: {response.status_code}")

def fetch_from_xml(file_path):
    """
    Fetch text from an XML document.

    Parameters:
    - file_path: str, the path to the XML document

    Returns:
    - str: The extracted text from the XML document
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    full_text = []

    for elem in root.iter():
        if elem.text:
            full_text.append(elem.text.strip())

    return ' '.join(full_text)

def fetch_from_word(file_path):
    """
    Fetch text from a Word document.

    Parameters:
    - file_path: str, the path to the Word document

    Returns:
    - str: The extracted text from the Word document
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Example usage:
# file_path = 'data/my_word_document.docx'
# word_text = fetch_from_word(file_path)
# Example usage:
# url = 'https://example.com/some-webpage'
# url_text = fetch_from_url(url)

def fetch_data(source_type, source_info, batch=1000):
    """
    Fetch data from Solr, a flat file, or a URL.

    Parameters:
    - source_type: str, one of 'solr', 'file', 'url'
    - source_info: depends on the source type:
        - for 'solr': source_info should be a dict with keys 'solr_url' and optionally 'query'
        - for 'file': source_info should be the file path as a string
        - for 'url': source_info should be the URL as a string
    """
    if source_type == 'solr':
        solr_url = source_info.get('solr_url')
        query = source_info.get('query', '*:*')  # Default query is '*:*'
        return fetch_from_solr(solr_url, query, batch)

    elif source_type == 'file':
        file_path = source_info
        return fetch_from_file(file_path)

    elif source_type == 'url':
        url = source_info
        return fetch_from_url(url)

    elif source_type == 'word':
        file_path = source_info
        return fetch_from_word(file_path)

    elif source_type == 'xml':
        file_path = source_info
        return fetch_from_xml(file_path)

    else:
       return None


def fetch_and_return_data(source_type, source_info):
    """
    Fetch data from the specified source and print it.

    Parameters:
    - source_type: str, one of 'solr', 'file', 'url'
    - source_info: depends on the source type:
        - for 'solr': source_info should be a dict with keys 'solr_url' and optionally 'query'
        - for 'file': source_info should be the file path as a string
        - for 'url': source_info should be the URL as a string
    """
    data = fetch_data(source_type, source_info)
    print(data)
    return data


'''    
# Example usage:
source_info_solr = {'solr_url': 'http://localhost:8983/solr/my_core', 'query': '*:*'}
solr_text = fetch_data('solr', source_info_solr)

source_info_file = 'data/my_text_file.txt'
file_text = fetch_data('file', source_info_file)

source_info_url = 'https://example.com/some-webpage'
url_text = fetch_data('url', source_info_url)

# Example usage:
source_info_solr = {'solr_url': 'http://localhost:8983/solr/my_core', 'query': '*:*'}
fetch_and_print_data('solr', source_info_solr)

source_info_file = 'data/my_text_file.txt'
fetch_and_print_data('file', source_info_file)

source_info_url = 'https://example.com/some-webpage'
fetch_and_print_data('url', source_info_url)

'''
