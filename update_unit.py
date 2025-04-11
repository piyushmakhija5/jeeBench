import os
import json
import base64
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from anthropic import Anthropic
import tqdm
# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def load_syllabus() -> Dict[str, Dict[str, str]]:
    """Load the JEE syllabus mapping"""
    with open("data/jee_syllabus.json", "r") as f:
        return json.load(f)

def extract_answer_keys(pdf_path: str) -> Dict[str, str]:
    """Extract answer keys from the answer key PDF file"""
    answers = {}
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Pattern to match question numbers and answers
        pattern = r"Q\.(\d+)\s*[:|.]\s*(\d+|\([A-D]\))"
        
        # Extract text from each page
        for page in doc:
            text = page.get_text()
            
            # Find all matches
            matches = re.finditer(pattern, text)
            for match in matches:
                question_num = int(match.group(1))
                answer = match.group(2)
                answers[question_num] = answer.strip('()')
                
        doc.close()
        return answers
        
    except Exception as e:
        print(f"Error extracting answer keys: {str(e)}")
        return {}

def analyze_image(image_path: str, syllabus: Dict[str, Dict[str, str]]) -> str:
    """Analyze an image using Claude to identify the unit"""
    
    # Read the image file
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Create the system prompt
    system_prompt = """
    You are an expert in analyzing JEE Advanced questions. Given an image of a JEE question:
    1. Identify the subject (Mathematics, Physics, or Chemistry)
    2. Based on the question content and the provided syllabus, identify which unit this question belongs to
    3. Return ONLY the unit name as listed in the syllabus, nothing else
    """
    
    # Create the message
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=100,
        temperature=0.0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Here is the syllabus mapping:\n{json.dumps(syllabus, indent=2)}\n\nPlease identify the unit for this question."
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text.strip()

def main():
    # Load the metadata and syllabus
    with open("extracted_questions/question_metadata.json", "r") as f:
        metadata = json.load(f)
    
    syllabus = load_syllabus()
    
    # Extract answer keys
    answer_keys = extract_answer_keys("data/answer_keys/JEE_advanced_2024_Paper_2_answers.pdf")
    
    # Process each question
    for question in tqdm.tqdm(metadata):
        try:
            # Get the unit from Claude
            unit = analyze_image(question["image_path"], syllabus)
            
            # Update the metadata with unit
            question["unit"] = unit
            
            # Add answer key if available
            if question["question_number"] in answer_keys:
                question["answer"] = answer_keys[question["question_number"]]
            
            print(f"Processed {question['image_path']}: {unit}")
            if question["question_number"] in answer_keys:
                print(f"Answer: {answer_keys[question['question_number']]}")
            
        except Exception as e:
            print(f"Error processing {question['image_path']}: {str(e)}")
    
    # Save the updated metadata
    with open("extracted_questions/question_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()