import requests
from bs4 import BeautifulSoup
import time
import os

# Base URL of the Stanford Encyclopedia of Philosophy
base_url = 'https://plato.stanford.edu/archives/fall2023'

# Function to get the list of all entries
def get_all_entry_urls():
    entries_index_url = f'{base_url}/contents.html'  # The URL where all entries are listed
    response = requests.get(entries_index_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links to entries based on page structure
    entries = soup.find_all('a', href=True)
    entry_urls = [base_url + '/' + entry['href'] for entry in entries if entry['href'].startswith('entries/')]
    
    return entry_urls
    
# Function to scrape an entry page
def scrape_entry(entry_url):
    response = requests.get(entry_url)
    
    if response.status_code != 200:
        return f"Error: Status code {response.status_code}", ''

    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', id='aueditable')
    if not content_div:
        return "Content div not found", ''
    text_content = content_div.get_text(separator='\n', strip=True)
    title = content_div.find('h1')
    if not title:
        return "Title not found within content div", ''
    title_text = title.get_text(strip=True)
    return title_text, text_content

# Function to save the text content to a file
def save_text_to_file(title, text_content):
    # Create the 'entries' directory if it does not exist
    entries_dir = os.path.join(os.getcwd(), 'entries')
    if not os.path.exists(entries_dir):
        os.makedirs(entries_dir)
    
    # Sanitize the file name by removing disallowed characters
    filename = ''.join(char for char in title if char.isalnum() or char in (' ', '-')).rstrip()
    
    # Specify the path to the file within the 'entries' directory
    filepath = os.path.join(entries_dir, f"{filename}.txt")
    
    # Write the text content to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(text_content)

# Main scraping logic
def main():
    entry_urls = get_all_entry_urls()
    
    for url in entry_urls:
        title, text_content = scrape_entry(url)
        print(f'Scraping and saving: {title}')
        
        # Save the text content to a file
        save_text_to_file(title, text_content)

        # Respect the crawl-delay of 5 seconds
        time.sleep(5)

if __name__ == '__main__':
    main()
