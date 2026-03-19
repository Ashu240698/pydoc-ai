"""
Data Collection Module

Downloads and collects Python documentation from multiple sources including
the Python standard library, requests library, and pandas library. This module
handles HTML fetching, parsing, and cleaning of documentation content for
later preprocessing and indexing.
"""
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import config

class DocCollector:
    """Collects and processes documentation from Python docs and libraries.

    This class orchestrates the downloading, parsing, and cleaning of
    documentation content from multiple sources. It uses requests for HTTP
    and BeautifulSoup for HTML parsing.
    """

    def __init__(self):
        """Initialize the documentation collector with proper configuration.

        Sets up output directory, HTTP session with appropriate headers,
        and configures the requests library for reliable document fetching.
        """
        # Create or verify raw data directory exists
        self.raw_dir = config.RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HTTP session with proper user agent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyDocAI/1.0 (Educational Project)'
        })
    
    def fetch_page(self, url):
        """Fetch a single HTTP page with comprehensive error handling.

        Makes HTTP requests with a 10-second timeout and appropriate
        error handling to gracefully manage network failures.

        Args:
            url (str): The URL to fetch.

        Returns:
            str or None: HTML content if successful, None on error.
        """
        try:
            # Make GET request with timeout
            response = self.session.get(url, timeout=10)
            # Raise exception for HTTP error codes
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            # Log error and return None for graceful failure
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def extract_text_from_html(self, html, url):
        """Extract clean, readable text from raw HTML documentation.

        Parses HTML with BeautifulSoup, removes non-content elements,
        finds main content area, and cleans up formatting for better
        text extraction and preprocessing.

        Args:
            html (str): Raw HTML content to parse.
            url (str): Original URL (for reference).

        Returns:
            str or None: Extracted and cleaned text, or None on failure.
        """
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Remove common non-content elements from the tree
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Find main content area - try multiple selectors for different sites
        main_content = (
            soup.find('div', class_='body') or  # Python docs
            soup.find('article') or              # Read the Docs
            soup.find('main') or                 # Generic
            soup.find('div', role='main') or
            soup.body
        )

        # Return None if no content found
        if not main_content:
            return None

        # Extract text with proper separator
        text = main_content.get_text(separator='\n', strip=True)

        # Clean up excessive whitespace and empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)

        return clean_text
    
    def collect_python_stdlib(self):
        """Collect documentation for Python standard library modules.

        Fetches documentation for a curated list of commonly-used stdlib
        modules from the official Python documentation website. Includes
        rate limiting to be respectful to the server.

        Returns:
            list: List of dictionaries with module documentation data.
        """
        print("\n📚 Collecting Python Standard Library docs...")

        # Define commonly-used stdlib modules to collect
        stdlib_modules = [
            'os', 'sys', 'json', 'csv', 'datetime', 'time',
            're', 'math', 'random', 'collections', 'itertools',
            'functools', 'pathlib', 'subprocess', 'argparse',
            'logging', 'unittest', 'urllib', 'http', 'email'
        ]

        # Base URL for Python documentation
        base_url = "https://docs.python.org/3/library/"
        docs = []

        # Fetch documentation for each module
        for module in tqdm(stdlib_modules, desc="Stdlib modules"):
            url = f"{base_url}{module}.html"
            html = self.fetch_page(url)

            # Process if fetch was successful
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'Python stdlib',
                        'module': module,
                        'url': url,
                        'text': text
                    })
                # Rate limiting - be polite to servers
                time.sleep(0.5)

        # Save collected documentation to JSON file
        output_file = self.raw_dir / 'python_stdlib.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)

        # Confirm successful collection
        print(f"✅ Collected {len(docs)} stdlib modules")
        return docs
    
    def collect_requests_docs(self):
        """Collect documentation from the requests library.

        Fetches documentation pages from the official requests library
        documentation site, covering installation, usage, and API reference.

        Returns:
            list: List of dictionaries with requests documentation data.
        """
        print("\n📚 Collecting requests library docs...")

        # Define documentation pages to collect
        pages = [
            'user/quickstart',
            'user/advanced',
            'api',
            'user/install',
            'user/authentication'
        ]

        # Base URL for requests documentation
        base_url = "https://requests.readthedocs.io/en/latest/"
        docs = []

        # Fetch documentation for each page
        for page in tqdm(pages, desc="requests docs"):
            url = f"{base_url}{page}.html"
            html = self.fetch_page(url)

            # Process if fetch was successful
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'requests',
                        'page': page,
                        'url': url,
                        'text': text
                    })
                # Rate limiting - be polite to servers
                time.sleep(0.5)

        # Save collected documentation to JSON file
        output_file = self.raw_dir / 'requests_docs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)

        # Confirm successful collection
        print(f"✅ Collected {len(docs)} requests pages")
        return docs
    
    def collect_pandas_docs(self):
        """Collect documentation from the pandas library.

        Fetches documentation pages from the official pandas documentation
        site, covering tutorials, guides, and API reference for commonly-used
        functions and classes.

        Returns:
            list: List of dictionaries with pandas documentation data.
        """
        print("\n📚 Collecting pandas docs...")

        # Define documentation pages to collect
        pages = [
            'getting_started/intro_tutorials/01_table_oriented',
            'getting_started/intro_tutorials/02_read_write',
            'getting_started/intro_tutorials/03_subset_data',
            'getting_started/intro_tutorials/04_plotting',
            'reference/api/pandas.DataFrame',
            'reference/api/pandas.read_csv',
            'reference/api/pandas.read_excel'
        ]

        # Base URL for pandas documentation
        base_url = "https://pandas.pydata.org/docs/"
        docs = []

        # Fetch documentation for each page
        for page in tqdm(pages, desc="pandas docs"):
            url = f"{base_url}{page}.html"
            html = self.fetch_page(url)

            # Process if fetch was successful
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'pandas',
                        'page': page,
                        'url': url,
                        'text': text
                    })
                # Rate limiting - be polite to servers
                time.sleep(0.5)

        # Save collected documentation to JSON file
        output_file = self.raw_dir / 'pandas_docs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)

        # Confirm successful collection
        print(f"✅ Collected {len(docs)} pandas pages")
        return docs
    
    def collect_all(self):
        """Orchestrate collection from all documentation sources.

        Coordinates the collection of documentation from Python stdlib,
        requests library, and pandas library. Aggregates all results
        and provides summary statistics.

        Returns:
            list: Combined list of all collected documentation dictionaries.
        """
        print("🚀 Starting documentation collection...")
        print(f"📁 Saving to: {self.raw_dir}")

        # Initialize list to store all collected documents
        all_docs = []

        # Collect documentation from all configured sources
        all_docs.extend(self.collect_python_stdlib())
        all_docs.extend(self.collect_requests_docs())
        all_docs.extend(self.collect_pandas_docs())

        # Display final statistics
        print(f"\n✅ Total documents collected: {len(all_docs)}")
        print(f"📁 Saved to: {self.raw_dir}")

        return all_docs


def main():
    """Main entry point for the data collection process.

    Initializes the DocCollector and orchestrates the complete
    documentation collection workflow.
    """
    # Create collector instance and start collection process
    collector = DocCollector()
    collector.collect_all()
    print("\n🎉 Data collection complete!")


if __name__ == "__main__":
    # Execute the main collection process when run directly
    main()