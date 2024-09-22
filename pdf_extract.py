import pdfplumber

import text_cleaner
from text_cleaner import clean_text, clean_pdf_text

# Path to the PDF file
pdf_path = "/Users/kevincowan/training_data/training_pdfs/Oxford History of World Cinema, The (Geoffrey Knowell-Smith).pdf"  # Replace with the actual path to your PDF


# Function to extract text from each page
def extract_text_from_pdf(pdf_path):
    all_text = ""

    # Open the PDF using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through all the pages
        for page_number, page in enumerate(pdf.pages):
            print(f"Extracting text from page {page_number + 1}...")
            # Extract the text from the page
            text = page.extract_text()
            if text:
                all_text += text
            else:
                print(f"No text found on page {page_number + 1}")
    # clean up text
    all_text = clean_text(all_text)
    return all_text


# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Optionally, save the extracted text to a file
with open("test/soft-skills.txt", "w", encoding="utf-8") as text_file:
    text_file.write(pdf_text)

print("Text extraction complete. Saved to 'solr_in_action_text.txt'.")
