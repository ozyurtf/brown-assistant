import tiktoken
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict

def chunk_bulletin(department: str, content: str) -> List[Tuple[str, Dict]]:
    """
    Split content by ### headers and create chunks with metadata.
    
    Args:
        department: The bulletin department (e.g., 'math', 'comp', etc.)
        content: The full content string
        
    Returns:
        List of tuples containing (chunk_text, metadata)
    """
    chunks = []
    metadata = []

    # Split by ### headers
    sections = re.split(r'^###\s*', content, flags=re.MULTILINE)
    
    # The first section is the main content (before any ### headers)
    if sections[0].strip():
        main_content = sections[0].strip()
        section_title =  'Main Content'
        chunk = f"{section_title}\n{main_content}"
        chunks.append(chunk)
        
        metadata.append({
            'department': department,
        })        
    
    # Process sections with ### headers
    for i, section in enumerate(sections[1:], 1):
        if section.strip():
            # Extract the header title (first line after ###)
            lines = section.strip().split('\n')
            header_title = lines[0].strip() if lines else f'Section {i}'
            section_content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
            
            # Only add if there's actual content
            if section_content:
                metadata.append({
                    'department': department,
                })
                section_title = header_title
                chunk = f"{section_title}\n{section_content}"
                chunks.append(chunk)
    
    return chunks, metadata

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def get_brown_concentrations():
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

def format_course(cab):
    course_info_all = []
    for term in cab.keys(): 
        for dept in cab[term].keys():
            for course in cab[term][dept]:
                output_lines = []
                output_lines.append(f"Term: {term}")
                output_lines.append(f"Department: {dept}")
                output_lines.append(f"Course ID: {course.get('course_id', '')}")
                output_lines.append(f"Course: {course.get('course', '')}")
                output_lines.append(f"Title: {course.get('title', '')}")
                output_lines.append(f"Total Sections: {course.get('total_sections', '')}")
                output_lines.append(f"Instructor Name: {course.get('instructor_name', '')}")
                output_lines.append(f"Instructor Email: {course.get('instructor_email', '')}")
                output_lines.append(f"Meeting Times: {course.get('meeting_times', '')}")
                output_lines.append(f"Description: {course.get('description', '')}")
                output_lines.append(f"Registration Restrictions: {course.get('registration_restrictions', '')}")
                output_lines.append(f"Course Attributes: {course.get('course_attributes', '')}")
                output_lines.append(f"Exam Info: {course.get('exam_info', '')}")
                output_lines.append(f"Class Notes: {course.get('class_notes', '')}")
                output_lines.append(f"Sections Text: {course.get('sections_text', '')}")
                course_info = "\n".join(output_lines)
                course_info_all.append(course_info)
    return course_info_all