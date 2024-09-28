import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse


# Function to crawl a website and extract text and links
def web_crawler(url, visited=None):
    if visited is None:
        visited = set()

    # Avoid revisiting the same page
    if url in visited:
        return

    try:
        # Step 2: Fetch the web page using the requests module
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve content. Status code: {response.status_code}")
            return None

        # Step 3: Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the relevant text content, usually found in <p> and heading tags
        page_text = []

        # Extract all paragraph text
        for paragraph in soup.find_all('p'):
            page_text.append(paragraph.get_text(strip=True))

        # Extract all headings (h1, h2, etc.)
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            page_text.append(heading.get_text(strip=True))

        # Combine all the extracted text into one large string
        text_content = "\n".join(page_text)

        # Save the URL and content to Solr
        extracted_data = {
            "id": url,
            "body": text_content
        }
        save_to_solr(extracted_data, solr_url, collection)

        # Mark the URL as visited
        visited.add(url)

        # Step 4: Extract all links and crawl them
        for link in soup.find_all('a', href=True):
            link_url = urljoin(url, link['href'])  # Resolve relative URLs

            # Ensure we only crawl within the same domain
            if urlparse(link_url).netloc == urlparse(url).netloc:
                web_crawler(link_url, visited)

    except Exception as e:
        print(f"An error occurred while crawling {url}: {e}")


# Function to save extracted data to Solr
def save_to_solr(data, solr_url, collection):
    # Solr update endpoint for adding documents
    solr_update_url = f"{solr_url}/{collection}/update/json/docs"

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send a POST request with the data in JSON format to Solr
        response = requests.post(solr_update_url, data=json.dumps(data), headers=headers)

        # Commit the changes to Solr (optional, depending on Solr settings)
        requests.get(f"{solr_url}/{collection}/update?commit=true")

        if response.status_code < 210:
            print(f"Document successfully added to Solr for URL: {data['id']}")
        else:
            print(f"Failed to add document to Solr. Status code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"An error occurred while saving to Solr: {e}")

if __name__ == "__main__":
    # Example usage
    url = 'https://huggingface.co/docs/huggingface_hub/guides/overview'  # Replace this with the target website URL
    solr_url = 'http://localhost:8983/solr'  # Replace with your Solr server URL
    collection = 'huggingface'  # Replace with your Solr collection name


    # Start crawling from the initial URL
    web_crawler(url)
