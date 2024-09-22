import re


def clean_text(text):
    # Remove website references
    text = re.sub(r'www\.\S+', '', text)

    # Remove page numbers and titles
    text = re.sub(r'\b\d{1,3}\b', '', text)  # Remove isolated numbers (page numbers)
    text = re.sub(r'(chapter|section|contents|index)\s*\d+', '', text, flags=re.IGNORECASE)  # Remove chapter numbers
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines

    # Remove headers, footers, and common irrelevant patterns
    text = re.sub(r'Soft Skills.*', '', text,
                  flags=re.IGNORECASE)  # Example for specific book titles (can adjust based on text)
    text = re.sub(r'MANNING\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'For online.*', '', text, flags=re.IGNORECASE)  # Remove publishing-related information
    text = re.sub(r'www.it-ebooks.info', '', text, flags=re.IGNORECASE)

    # Remove Table of Contents (or patterns related to it)
    text = re.sub(r'contents\s*(section|part|chapter)\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove unwanted text blocks
    toc_pattern = r'(brief contents|contents|index|foreword|preface|acknowledgments|about this book)\s*.*?\s*\d+'
    text = re.sub(toc_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove extra spaces and unnecessary blank lines
    text = re.sub(r'\n\s*\n', '\n', text)  # Clean multiple newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    return text


import re


def clean_pdf_text(text):
    """
    Cleans the raw text extracted from a PDF document by removing page numbers, TOC,
    and other common errata like headers or footers.

    Args:
        text (str): The raw text extracted from a PDF document.

    Returns:
        str: Cleaned body text.
    """
    # Remove page numbers (e.g., 'Page 1', '1', '[1]', '1/100')
    text = re.sub(r'(\bPage\b\s?\d+|\[\d+\]|\b\d{1,3}\b\s?/\s?\d{1,3})', '', text)

    # Remove TOC entries (common patterns, you can adjust based on your TOC format)
    text = re.sub(r'(\.{2,}|\s{2,})\d{1,3}', '', text)  # Matches '....123' or large spaces before page numbers in TOC

    # Remove any line that looks like a header or footer (e.g., repetitive patterns like chapter titles or document titles)
    text = re.sub(
        r'(^Chapter\s\d+.*|^Table of Contents.*|^Copyright.*|^www\..*|^http[s]?://.*|^[0-9A-Za-z-]+\s+by\s+.*)', '',
        text, flags=re.MULTILINE)

    # Remove extra newlines or whitespace
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces

    # Remove leftover unwanted formatting
    text = text.strip()

    return text




if __name__ == "__main__":
    # Path to the raw text file extracted from the PDF
    file_path = 'test/soft-skills.txt'

    # Read the raw extracted text
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Save the cleaned text to a new file
    output_file_path =  'test/soft-skills.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(cleaned_text)

    print("Text cleaning completed and saved to 'soft-skills-cleaned.txt'.")

    # Example usage
    raw_text = """Page 1

        Chapter 1: Introduction............................1
        Chapter 2: Overview................................5
        www.example.com

        This is the body text of the document. It continues here without any page numbers or headers.

        Page 2

        And this is the continuation of the body text."""

    cleaned_text = clean_pdf_text(raw_text)
    print(cleaned_text)
