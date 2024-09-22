import os
import requests
from bs4 import BeautifulSoup
import time

from extractors.loyalbooks import TRAINING_DATA_PATH

# Base URL for LoyalBooks
BASE_URL = "https://www.loyalbooks.com"

# Directory to save training data
V1_TRAINING_DATA_PATH = "/Users/kevincowan/training_data"

TRAINING_DATA_PATH = "/Users/kevincowan/training_data_2"

PREVIOUS_TRAINING_DATA_PATHS = [
    V1_TRAINING_DATA_PATH,
]

# Create the training data directory if it doesn't exist
if not os.path.exists(TRAINING_DATA_PATH):
    os.makedirs(TRAINING_DATA_PATH)


def get_soup(url):
    """
    Helper function to get BeautifulSoup object for a given URL
    """
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    return None


def extract_books_from_page(page_url):
    """
    Extracts book links from the LoyalBooks main page
    """
    # check the url for a 404, and return if true
    response = requests.get(page_url)
    if response.status_code == 404:
        print("End of line..." + page_url)
        return []

    soup = get_soup(page_url)

    if not soup:
        return []

    # Find all the book links from elements with class "innertube"
    # Find the table with the attribute summary="Audio Books"
    tables = soup.find_all('table')

    # Look for the table with the attribute summary="Audio Books"
    table = None
    for tbl in tables:
        if tbl.attrs.get('summary', '').lower() == 'audio books':
            table = tbl
            break

    # Extract all hrefs from the table
    book_links = []
    if table:
        for link in table.find_all('a', href=True):
            href = link.get('href')
            if href and href.startswith('/'):
                book_links.append(BASE_URL + "/" + href.strip('/'))
                print("Add: " + BASE_URL + "/" + href.strip('/'))

    return book_links


def download_book_text(book_url):
    """
    Given a book page URL, download the text file if available
    """
    soup = get_soup(book_url)

    if not soup:
        return None

    # Extract the href for the text file download link
    text_download_link = None
    download_container = soup.find('div', {'class': 's-book', 'id': 'text'})

    if download_container:
        download_link_tag = download_container.find_parent('a', href=True)
        if download_link_tag:
            text_download_link = download_link_tag['href']

    if not text_download_link:
        return None

    # Form the full URL for the text download
    if text_download_link.startswith('/'):
        text_download_link = BASE_URL.rstrip('/') + text_download_link

    # Get the filename from the URL
    file_name = os.path.basename(text_download_link)

    # Check if the file already exists in any of the previous training data directories
    for previous_path in PREVIOUS_TRAINING_DATA_PATHS:
        if os.path.exists(os.path.join(previous_path, file_name)):
            print(f"Skipping {file_name}, already downloaded in {previous_path}.")
            return None

    # Check if the file already exists in the training data directory
    if os.path.exists(os.path.join(TRAINING_DATA_PATH, file_name)):
        print(f"Skipping {file_name}, already downloaded.")
        return None

    # Download the text file
    print(f"Downloading {file_name} from {text_download_link}...")
    response = requests.get(text_download_link)
    if response.status_code == 200:
        # Save the file
        file_path = os.path.join(TRAINING_DATA_PATH, file_name)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Saved {file_name} to {TRAINING_DATA_PATH}")
        return file_path
    else:
        print(f"Failed to download {file_name} from {text_download_link}.")
    return None


def main_loop():
    """
    Main loop to extract and download books from LoyalBooks
    """
    page_number = 2

    while True:
        if page_number > 1:
            page_url = f"{BASE_URL}/genre/Historical_Fiction?results=100&Page={page_number}"
        else:
            page_url = BASE_URL + "/genre/Historical_Fiction?results=100"
        print(f"Extracting books from {page_url}...")
        book_links = extract_books_from_page(page_url)
        if not book_links:
            print(f"No book links found on page {page_number}. Stopping.")
            break

        for book_url in book_links:
            download_book_text(book_url)

        page_number += 1
        print(f"Moving to page {page_number}...")

if __name__ == "__main__":
    main_loop()

