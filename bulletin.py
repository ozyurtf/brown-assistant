import asyncio
import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Tuple

def get_bulletin_codes():
   url = "https://bulletin.brown.edu/the-college/concentrations/"
   response = requests.get(url)
   soup = BeautifulSoup(response.content, 'html.parser')
   
   concentration_links = soup.find_all('a', href=re.compile(r'/the-college/concentrations/[a-zA-Z]+/$'))
   
   concentration_codes = []
   for link in concentration_links:
       href = link.get('href')
       match = re.search(r'/the-college/concentrations/([a-zA-Z]+)/', href)
       if match:
           concentration_codes.append(match.group(1))
   
   return sorted(list(set(concentration_codes)))    

def clean_content(content):
    """
    Extract main academic content by removing navigation and UI elements
    """
    content = re.sub(r'^.*?(?=\n# [A-Z])', '', content, flags=re.DOTALL)
    content = re.sub(r'- \[.*?\]\(.*?\).*?(?=\n[^-])', '', content, flags=re.DOTALL)
    content = re.sub(r'\n  \* \*\*Your Concentration\(s\)\*\*.*?(?=\n\n|\n[A-Z])', '', content, flags=re.DOTALL)
    content = re.sub(r'\n  \* \[Foreword\].*?(?=\n\n|\n[A-Z])', '', content, flags=re.DOTALL)
    content = re.sub(r'\n#### Brown University.*$', '', content, flags=re.DOTALL)

    print_option_patterns = [
        r'^\[\s*Print Options\s*\]\(.*?\)\s*\n?',
        r'^\s*Print Options\s*$\n?',
        r'^\[\s*Send Page to Printer\s*\]\(.*?\)\s*\n?',
        r'^_\s*Print this page\.\s*_\s*\n?',
        r'^_\s*The PDF will include all information unique to this page\.\s*_\s*\n?',
        r'^\[\s*Download PDF of this page\s*\]\(.*?\)\s*\n?',
        r'^\s*Download Complete PDFs\s*$\n?',
        r'^\[\s*Cancel\s*\]\(.*?\)\s*\n?',
    ]
    for pattern in print_option_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)

    patterns_to_remove = [
        r'\[Toggle Navigation\].*?\n',
        r'\[Back to top\].*?\n',
        r'Search Bulletin.*?\n',
        r'Login.*?\n',
        r'You\'re logged in.*?\n'
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    content = re.sub(r'https?://[^\s\n]+', '', content)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = content.strip()

    return content

async def scrape_all_concentrations(urls):
    """
    Scrape all Brown concentration pages in parallel and return cleaned content.
    
    Args:
        urls: List of concentration URLs to scrape
        
    Returns:
        dict: Dictionary with concentration codes as keys and cleaned content as values
    """
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        verbose=False
    )

    bulletin = {}
    
    async with AsyncWebCrawler() as crawler:
        print(f"Starting to scrape {len(urls)} concentration pages...")
        results = await crawler.arun_many(urls, config=run_conf)
        
        successful = 0
        failed = 0
        
        for res in results:
            if res.success:
                try:
                    cleaned_content = clean_content(res.markdown.raw_markdown)
                    code = res.url.split("/")[-2]
                    bulletin[code] = cleaned_content
                    successful += 1
                    print(f"{code}: {len(res.markdown.raw_markdown)} chars")
                except Exception as e:
                    failed += 1
                    print(f"Error processing {res.url}: {e}")
            else:
                failed += 1
                print(f"Failed to fetch {res.url}: {res.error_message}")
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
        
    return bulletin

def save_bulletin_to_json(bulletin, filename="files/bulletin.json"):
    """Save bulletin dictionary to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(bulletin, f, indent=2, ensure_ascii=False)
        print(f"Saved bulletin data to {filename}")
        print(f"Total concentrations: {len(bulletin)}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

def main():
    """Main function to scrape concentrations and save to JSON."""
    codes = get_bulletin_codes()
    urls = [f"https://bulletin.brown.edu/the-college/concentrations/{code}/" for code in codes]
    print(f"Found {len(codes)} concentration codes")
    bulletin = asyncio.run(
        scrape_all_concentrations(urls)
    )
    save_bulletin_to_json(bulletin)

if __name__ == "__main__":
    main()   