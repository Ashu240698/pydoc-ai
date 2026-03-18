"""
Data Collection Module
Downloads Python documentation from various sources.
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
    """Collects documentation from Python docs and libraries."""
    
    def __init__(self):
        self.raw_dir = config.RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyDocAI/1.0 (Educational Project)'
        })
    
    def fetch_page(self, url):
        """Fetch a single page with error handling."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def extract_text_from_html(self, html, url):
        """Extract clean text from HTML documentation."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, nav elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Get main content (varies by site)
        main_content = (
            soup.find('div', class_='body') or  # Python docs
            soup.find('article') or              # Read the Docs
            soup.find('main') or                 # Generic
            soup.find('div', role='main') or
            soup.body
        )
        
        if not main_content:
            return None
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive newlines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        return clean_text
    
    def collect_python_stdlib(self):
        """Collect Python standard library documentation."""
        print("\n📚 Collecting Python Standard Library docs...")
        
        # Key stdlib modules to collect
        stdlib_modules = [
            'os', 'sys', 'json', 'csv', 'datetime', 'time',
            're', 'math', 'random', 'collections', 'itertools',
            'functools', 'pathlib', 'subprocess', 'argparse',
            'logging', 'unittest', 'urllib', 'http', 'email'
        ]
        
        base_url = "https://docs.python.org/3/library/"
        docs = []
        
        for module in tqdm(stdlib_modules, desc="Stdlib modules"):
            url = f"{base_url}{module}.html"
            html = self.fetch_page(url)
            
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'Python stdlib',
                        'module': module,
                        'url': url,
                        'text': text
                    })
                time.sleep(0.5)  # Be polite to servers
        
        # Save to file
        output_file = self.raw_dir / 'python_stdlib.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Collected {len(docs)} stdlib modules")
        return docs
    
    def collect_requests_docs(self):
        """Collect requests library documentation."""
        print("\n📚 Collecting requests library docs...")
        
        pages = [
            'user/quickstart',
            'user/advanced',
            'api',
            'user/install',
            'user/authentication'
        ]
        
        base_url = "https://requests.readthedocs.io/en/latest/"
        docs = []
        
        for page in tqdm(pages, desc="requests docs"):
            url = f"{base_url}{page}.html"
            html = self.fetch_page(url)
            
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'requests',
                        'page': page,
                        'url': url,
                        'text': text
                    })
                time.sleep(0.5)
        
        # Save to file
        output_file = self.raw_dir / 'requests_docs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Collected {len(docs)} requests pages")
        return docs
    
    def collect_pandas_docs(self):
        """Collect pandas documentation."""
        print("\n📚 Collecting pandas docs...")
        
        pages = [
            'getting_started/intro_tutorials/01_table_oriented',
            'getting_started/intro_tutorials/02_read_write',
            'getting_started/intro_tutorials/03_subset_data',
            'getting_started/intro_tutorials/04_plotting',
            'reference/api/pandas.DataFrame',
            'reference/api/pandas.read_csv',
            'reference/api/pandas.read_excel'
        ]
        
        base_url = "https://pandas.pydata.org/docs/"
        docs = []
        
        for page in tqdm(pages, desc="pandas docs"):
            url = f"{base_url}{page}.html"
            html = self.fetch_page(url)
            
            if html:
                text = self.extract_text_from_html(html, url)
                if text:
                    docs.append({
                        'source': 'pandas',
                        'page': page,
                        'url': url,
                        'text': text
                    })
                time.sleep(0.5)
        
        # Save to file
        output_file = self.raw_dir / 'pandas_docs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Collected {len(docs)} pandas pages")
        return docs
    
    def collect_all(self):
        """Collect all documentation."""
        print("🚀 Starting documentation collection...")
        print(f"📁 Saving to: {self.raw_dir}")
        
        all_docs = []
        
        # Collect from all sources
        all_docs.extend(self.collect_python_stdlib())
        all_docs.extend(self.collect_requests_docs())
        all_docs.extend(self.collect_pandas_docs())
        
        print(f"\n✅ Total documents collected: {len(all_docs)}")
        print(f"📁 Saved to: {self.raw_dir}")
        
        return all_docs


def main():
    """Main execution."""
    collector = DocCollector()
    collector.collect_all()
    print("\n🎉 Data collection complete!")


if __name__ == "__main__":
    main()