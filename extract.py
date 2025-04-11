import fitz  # PyMuPDF
import re
import os
import json
import tqdm
from collections import defaultdict

def extract_questions_from_pdf(pdf_path, output_folder, syllabus_path):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the syllabus
    try:
        with open(syllabus_path, 'r') as file:
            syllabus = json.load(file)
    except Exception as e:
        print(f"Error loading syllabus: {str(e)}")
        return
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {str(e)}")
        return
    
    # Find subjects in the document
    subjects = ["Mathematics", "Physics", "Chemistry"]
    subject_pages = defaultdict(list)
    
    # First pass: identify subjects and their pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        for subject in subjects:
            if subject in text:
                subject_pages[subject].append(page_num)
    
    # Refine subject page assignments (ensure consecutive pages)
    subject_ranges = {}
    if len(subject_pages) == 3:  # All three subjects found
        # Sort subjects by their first appearance
        sorted_subjects = sorted(subjects, key=lambda s: min(subject_pages[s]) if subject_pages[s] else float('inf'))
        
        for i, subject in enumerate(sorted_subjects):
            if i < len(sorted_subjects) - 1:
                start_page = min(subject_pages[subject])
                end_page = min(subject_pages[sorted_subjects[i+1]]) - 1
                subject_ranges[subject] = (start_page, end_page)
            else:
                start_page = min(subject_pages[subject])
                end_page = len(doc) - 1
                subject_ranges[subject] = (start_page, end_page)
    else:
        # Fallback if not all subjects found
        print("Warning: Not all subjects detected. Using default page mapping.")
        page_count = len(doc)
        per_subject = page_count // 3
        subject_ranges = {
            "Mathematics": (0, per_subject - 1),
            "Physics": (per_subject, 2 * per_subject - 1),
            "Chemistry": (2 * per_subject, page_count - 1)
        }
    
    # Define patterns for different question formats in JEE papers
    question_patterns = [
        r"Q\.(\d+)",           # Standard question format: Q.1, Q.2, etc.
        r"QUESTION\s+(\d+)",   # Alternate format: QUESTION 1, QUESTION 2, etc.
    ]
    
    # Dictionary to store question metadata
    question_metadata = []
    
    # Process each subject
    for subject, (start_page, end_page) in subject_ranges.items():
        print(f"Processing {subject} (pages {start_page+1}-{end_page+1})")
        
        # Extract questions for this subject
        questions = []
        
        # Find questions in subject pages
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]
            
            # Get text with positions
            text_instances = page.get_text("dict")["blocks"]
            
            for block in text_instances:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = " ".join([span["text"] for span in line["spans"]])
                        
                        # Check for question patterns
                        for pattern in question_patterns:
                            matches = re.finditer(pattern, line_text)
                            for match in matches:
                                # Extract y-position and question number
                                y_pos = line["bbox"][1]  # Top y-coordinate
                                
                                try:
                                    q_num = int(match.group(1))
                                    
                                    questions.append({
                                        "text": line_text.strip(),
                                        "page": page_num,
                                        "y_pos": y_pos,
                                        "q_num": q_num,
                                        "subject": subject
                                    })
                                except (IndexError, ValueError) as e:
                                    print(f"Error parsing question: {str(e)}")
        
        # Sort questions by page and y-position
        questions.sort(key=lambda q: (q["page"], q["y_pos"]))
        
        # Extract each question
        for i, question in enumerate(questions):
            start_page = question["page"]
            start_y = question["y_pos"] - 20  # Add margin above
            
            # Determine end position
            if i < len(questions) - 1 and questions[i+1]["page"] == start_page:
                end_y = questions[i+1]["y_pos"] - 10
            else:
                # Last question on page
                end_y = doc[start_page].rect.height
            
            # Extract and save the image
            page = doc[start_page]
            rect = fitz.Rect(0, start_y, page.rect.width, end_y)
            
            zoom = 2
            matrix = fitz.Matrix(zoom, zoom)
            try:
                pix = page.get_pixmap(clip=rect, matrix=matrix)
                
                # Create filename with subject and question number
                q_identifier = str(question["q_num"]).zfill(2)
                img_path = os.path.join(output_folder, f"{subject}_Q{q_identifier}.png")
                pix.save(img_path)
                
                print(f"Extracted {subject} Question {q_identifier} from page {start_page+1}")
                
                # Determine unit from syllabus
                question_text = question["text"]
                unit = determine_unit(question_text, syllabus[subject])
                
                # Add to metadata
                question_metadata.append({
                    "subject": subject,
                    "question_number": question["q_num"],
                    "page_number": start_page + 1,
                    "unit": unit,
                    "image_path": img_path
                })
            except Exception as e:
                print(f"Error extracting question image: {str(e)}")
            
            # Handle multi-page questions (simplified)
            if i < len(questions) - 1 and questions[i+1]["page"] > start_page:
                # Just extract the next pages up to the next question
                next_page = questions[i+1]["page"]
                for p in range(start_page + 1, next_page):
                    try:
                        page = doc[p]
                        pix = page.get_pixmap(matrix=matrix)
                        img_path = os.path.join(output_folder, f"{subject}_Q{q_identifier}_page_{p+1}.png")
                        pix.save(img_path)
                        print(f"Extracted {subject} Question {q_identifier} continuation from page {p+1}")
                    except Exception as e:
                        print(f"Error extracting continuation image: {str(e)}")
    
    # Extract section headers (optional)
    try:
        extract_sections(doc, output_folder)
    except Exception as e:
        print(f"Error extracting sections: {str(e)}")
    
    # Save metadata to JSON
    try:
        metadata_path = os.path.join(output_folder, "question_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(question_metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
    
    doc.close()
    print(f"Extracted {len(question_metadata)} questions to {output_folder}")

def determine_unit(question_text, subject_units):
    """
    Try to determine which unit the question belongs to based on keywords.
    Returns the most likely unit or "Unknown" if can't determine.
    """
    # Simple keyword matching
    max_matches = 0
    best_unit = "Unknown"
    
    for unit_name, unit_title in subject_units.items():
        # Convert to lowercase and split into keywords
        keywords = unit_title.lower().split()
        matches = sum(1 for keyword in keywords if keyword.lower() in question_text.lower())
        
        if matches > max_matches:
            max_matches = matches
            best_unit = f"{unit_name}: {unit_title}"
    
    return best_unit

def extract_sections(doc, output_folder):
    """Extract entire sections like SECTION 1, SECTION 2, etc."""
    
    section_pattern = r"SECTION\s+(\d+)"
    sections = []
    
    # Find section headers
    for page_num in tqdm.tqdm(range(len(doc)), desc="Finding sections"):
        page = doc[page_num]
        text = page.get_text("text")
        matches = re.finditer(section_pattern, text)
        
        for match in matches:
            section_num = match.group(1)
            # Find the text position
            text_instances = page.get_text("dict")["blocks"]
            for block in text_instances:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = " ".join([span["text"] for span in line["spans"]])
                        if f"SECTION {section_num}" in line_text:
                            sections.append({
                                "num": section_num,
                                "page": page_num,
                                "y_pos": line["bbox"][1]
                            })
    
    # Sort sections by page and position
    sections.sort(key=lambda s: (s["page"], s["y_pos"]))
    
    # Extract each section
    for i, section in enumerate(sections):
        try:
            start_page = section["page"]
            start_y = section["y_pos"] - 30  # Margin above
            
            # Determine end of section
            if i < len(sections) - 1:
                end_page = sections[i+1]["page"]
                end_y = sections[i+1]["y_pos"] - 10 if end_page == start_page else doc[start_page].rect.height
            else:
                # Last section goes to end of document
                end_page = len(doc) - 1
                end_y = doc[start_page].rect.height
            
            # Extract and save the section
            page = doc[start_page]
            rect = fitz.Rect(0, start_y, page.rect.width, end_y)
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            img_path = os.path.join(output_folder, f"Section_{section['num']}.png")
            pix.save(img_path)
            
            # Handle multi-page sections if needed
            if start_page != end_page:
                for p in range(start_page + 1, end_page + 1):
                    page = doc[p]
                    y_start = 0
                    y_end = page.rect.height if p < end_page else end_y
                    rect = fitz.Rect(0, y_start, page.rect.width, y_end)
                    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                    img_path = os.path.join(output_folder, f"Section_{section['num']}_page_{p+1}.png")
                    pix.save(img_path)
        except Exception as e:
            print(f"Error extracting section {section['num']}: {str(e)}")

# Usage
pdf_path = "data/question_papers/jee_advanced/JEE_Advanced_2024_Paper_2.pdf"
output_folder = "extracted_questions"
syllabus_path = "data/jee_syllabus.json"
extract_questions_from_pdf(pdf_path, output_folder, syllabus_path)