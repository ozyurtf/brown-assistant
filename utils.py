import tiktoken
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain.evaluation import load_evaluator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import evaluate

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def map_code_to_dept_cab():
    """Scrape the main page to find department codes"""
    url = "https://cab.brown.edu/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    departments = {}
    selects = soup.find_all('select')
    
    for select in selects:
        if 'dept' in str(select).lower():
            options = select.find_all('option')
            for option in options:
                if option.get('value') and len(option.get('value')) > 1:
                    departments[option.text.strip()] = option.get('value')
    
    return departments

def map_code_to_dept_bulletin():
    """Scrape Brown bulletin to get department codes and names"""
    url = "https://bulletin.brown.edu/the-college/concentrations/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    departments = {}
    
    # Look for links to concentration pages
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link.get('href')
        if href and '/concentrations/' in href and href.endswith('/'):
            # Extract department code from URL
            match = re.search(r'/concentrations/([a-zA-Z]+)/?$', href)
            if match:
                dept_code = match.group(1)
                dept_name = link.text.strip()
                
                # Skip empty names or navigation links
                if dept_name and len(dept_name) > 2 and not dept_name.lower() in ['home', 'back', 'next']:
                    departments[dept_name] = dept_code
    
    return departments

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

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


def compute_bleu_score(prediction: str, reference: str) -> float:
    """
    Compute BLEU score using Hugging Face's evaluate library.
    Args:
        prediction (str): model-generated text
        reference (str): ground-truth text
    Returns:
        float: BLEU score (0–1 range)
    """
    if not prediction or not reference:
        return 0.0

    result = bleu_metric.compute(
        predictions=[prediction],
        references=[[reference]]
    )
    return float(result["bleu"])


def compute_rouge_score(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score using Hugging Face's evaluate library.
    Args:
        prediction (str): model-generated text
        reference (str): ground-truth text
    Returns:
        float: ROUGE-L F1 score (0–1 range)
    """
    if not prediction or not reference:
        return 0.0

    result = rouge_metric.compute(
        predictions=[prediction],
        references=[reference]
    )
    return float(result["rougeL"])
