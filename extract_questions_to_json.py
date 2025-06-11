"""
Advanced extraction for 2025 JEE papers using question snapshots + AI analysis
Based on the original extract.py approach but enhanced for papers with embedded solutions
"""
import fitz  # PyMuPDF
import re
import os
import json
import base64
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import tqdm
from dotenv import load_dotenv

from anthropic import Anthropic
from logger import get_logger, setup_logging
from config import config

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(
    level=config.logging["level"],
    log_format=config.logging["log_format"],
    log_file=config.logging["log_file"],
    console=config.logging["console"]
)
logger = get_logger(__name__)


class AdvancedJEE2025Extractor:
    """Advanced extractor using question snapshots + AI analysis"""

    def __init__(self, pdf_path: str, output_folder: str, syllabus_path: str, scoring_path: str = None, log_file: Optional[str] = None):
        self.pdf_path = Path(pdf_path)
        self.output_folder = Path(output_folder)
        self.syllabus_path = Path(syllabus_path)
        self.scoring_path = Path(scoring_path) if scoring_path else Path("data/inputs/scoring/jee_2025_scoring.json")

        # Create output folder
        self.output_folder.mkdir(exist_ok=True)

        # Load syllabus
        self.syllabus = self._load_syllabus()

        # Load scoring configuration
        self.scoring_config = self._load_scoring_config()

        # Initialize Anthropic client for AI analysis
        self.anthropic_client = None
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = Anthropic(api_key=api_key)
                logger.info("Anthropic client initialized for AI analysis")
            else:
                logger.warning("No Anthropic API key found - AI analysis disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")

        # Strict question patterns (from original extract.py)
        self.question_patterns = [
            r"Q\.(\d+)",           # Q.1, Q.2, etc.
            r"QUESTION\s+(\d+)",   # QUESTION 1, QUESTION 2, etc.
        ]

        self.log_file = log_file or "outputs/logs/jeeBench.log"

        logger.info(f"Initialized advanced extractor for {self.pdf_path}")

    def _load_syllabus(self) -> Dict:
        """Load the JEE syllabus mapping"""
        try:
            with open(self.syllabus_path, 'r') as f:
                syllabus = json.load(f)
            logger.info(f"Loaded syllabus from {self.syllabus_path}")
            return syllabus
        except Exception as e:
            logger.error(f"Failed to load syllabus: {e}")
            return {}

    def _load_scoring_config(self) -> Dict:
        """Load the JEE scoring configuration"""
        try:
            with open(self.scoring_path, 'r') as f:
                scoring_config = json.load(f)
            logger.info(f"Loaded scoring config from {self.scoring_path}")
            return scoring_config
        except Exception as e:
            logger.error(f"Failed to load scoring config: {e}")
            return {}

    def _normalize_question_type(self, ai_question_type: str) -> str:
        """Normalize question type based on scoring configuration"""
        if not self.scoring_config:
            return ai_question_type

        # Valid question types from scoring config
        valid_types = list(self.scoring_config.keys())

        # Direct match
        if ai_question_type in valid_types:
            return ai_question_type

        # Fuzzy matching for common variations
        type_mapping = {
            "MCQ": "MCQ",
            "Single Correct": "MCQ",
            "Multiple Choice": "MCQ",
            "Multiple Correct": "Multiple Correct",
            "Multiple Select": "Multiple Correct",
            "Numerical": "Numerical",
            "Numerical Answer": "Numerical",
            "Integer": "Numerical",
            "Pair Matching": "Pair Matching",
            "Matrix Match": "Pair Matching",
            "Unknown": "MCQ"  # Default fallback
        }

        normalized_type = type_mapping.get(ai_question_type, "MCQ")

        if normalized_type != ai_question_type:
            logger.debug(f"Normalized question type '{ai_question_type}' to '{normalized_type}'")

        return normalized_type

    def extract_questions_and_answers(self) -> List[Dict]:
        """Extract questions using the original approach + AI analysis"""
        try:
            doc = fitz.open(self.pdf_path)
            logger.info(f"Opened PDF with {len(doc)} pages")

            # Step 1: Find subjects and their page ranges (from original extract.py)
            subject_ranges = self._find_subject_ranges(doc)

            # Step 2: Extract questions using original method
            all_questions_data = []

            for subject, (start_page, end_page) in subject_ranges.items():
                logger.info(f"Processing {subject} (pages {start_page+1}-{end_page+1})")

                # Find question locations (like original extract.py)
                questions = self._find_questions_in_subject(doc, subject, start_page, end_page)

                # Validate question count
                if len(questions) > 25 or len(questions) < 10:
                    logger.warning(f"{subject}: Found {len(questions)} questions - seems wrong!")
                else:
                    logger.info(f"{subject}: Found {len(questions)} questions - looks good")

                # Extract question images and analyze with AI
                for i, question in enumerate(tqdm.tqdm(questions, desc=f"Extracting {subject}")):
                    question_data = self._extract_and_analyze_question(doc, question, i, questions)
                    if question_data:
                        all_questions_data.append(question_data)

            doc.close()

            logger.info(f"Successfully extracted {len(all_questions_data)} questions total")
            return all_questions_data

        except Exception as e:
            logger.error(f"Failed to extract questions: {e}")
            raise

    def _find_subject_ranges(self, doc: fitz.Document) -> Dict[str, Tuple[int, int]]:
        """Find subjects and their page ranges (from original extract.py)"""
        subjects = ["Mathematics", "Physics", "Chemistry"]
        subject_pages = defaultdict(list)

        # First pass: identify subjects and their pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            for subject in subjects:
                if subject in text:
                    subject_pages[subject].append(page_num)

        # Refine subject page assignments (from original extract.py)
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
            logger.warning("Not all subjects detected. Using default page mapping.")
            page_count = len(doc)
            per_subject = page_count // 3
            subject_ranges = {
                "Mathematics": (0, per_subject - 1),
                "Physics": (per_subject, 2 * per_subject - 1),
                "Chemistry": (2 * per_subject, page_count - 1)
            }

        logger.info(f"Subject ranges: {subject_ranges}")
        return subject_ranges

    def _find_questions_in_subject(self, doc: fitz.Document, subject: str,
                                  start_page: int, end_page: int) -> List[Dict]:
        """Find questions in subject pages (from original extract.py)"""
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
                        for pattern in self.question_patterns:
                            matches = re.finditer(pattern, line_text)
                            for match in matches:
                                # Extract y-position and question number
                                y_pos = line["bbox"][1]  # Top y-coordinate

                                try:
                                    q_num = int(match.group(1))

                                    # Only accept reasonable question numbers (1-20)
                                    if 1 <= q_num <= 20:
                                        questions.append({
                                            "text": line_text.strip(),
                                            "page": page_num,
                                            "y_pos": y_pos,
                                            "q_num": q_num,
                                            "subject": subject
                                        })
                                except (IndexError, ValueError):
                                    continue

        # Sort questions by page and y-position (from original extract.py)
        questions.sort(key=lambda q: (q["page"], q["y_pos"]))

        # Remove duplicates (same question number)
        seen_nums = set()
        unique_questions = []
        for q in questions:
            if q["q_num"] not in seen_nums:
                unique_questions.append(q)
                seen_nums.add(q["q_num"])

        return unique_questions

    def _extract_and_analyze_question(self, doc: fitz.Document, question: Dict,
                                    index: int, all_questions: List[Dict]) -> Optional[Dict]:
        """Extract question image and analyze with AI (enhanced from original)"""
        try:
            # Extract question image (from original extract.py)
            start_page = question["page"]
            start_y = question["y_pos"] - config.processing.question_margin_above

            # Determine end position
            if index < len(all_questions) - 1 and all_questions[index + 1]["page"] == start_page:
                end_y = all_questions[index + 1]["y_pos"] - config.processing.question_margin_below
            else:
                # Last question on page
                end_y = doc[start_page].rect.height

            # Extract and save the image
            page = doc[start_page]
            rect = fitz.Rect(0, start_y, page.rect.width, end_y)

            matrix = fitz.Matrix(config.processing.pdf_zoom, config.processing.pdf_zoom)
            pix = page.get_pixmap(clip=rect, matrix=matrix)

            # Create filename with subject and question number
            q_identifier = str(question["q_num"]).zfill(2)
            img_path = self.output_folder / f"{question['subject']}_Q{q_identifier}_2025.png"
            pix.save(str(img_path))

            logger.debug(f"Extracted {question['subject']} Question {q_identifier} image")

            # Analyze with AI if available
            analysis_result = {}
            if self.anthropic_client:
                analysis_result = self._analyze_question_with_ai(img_path, question["subject"])

            # Normalize question type using scoring config
            raw_question_type = analysis_result.get("type", "Unknown")
            normalized_question_type = self._normalize_question_type(raw_question_type)

            # Use unit directly from AI analysis (no normalization needed)
            unit = analysis_result.get("unit", "Unknown")

            # Add scoring information based on question type
            scoring_info = self.scoring_config.get(normalized_question_type, {})

            # Create simplified question metadata (removing duplicate fields)
            question_data = {
                "subject": question["subject"],
                "question_number": question["q_num"],
                "page_number": start_page + 1,
                "unit": unit,
                "image_path": str(img_path),
                "answer_key": analysis_result.get("answer", None),
                "choices": analysis_result.get("choices", []),
                "question_type": normalized_question_type,
                "scoring": {
                    "full_marks": scoring_info.get("full_marks", 0),
                    "negative_marks": scoring_info.get("negative_marks", 0),
                    "partial_marks": scoring_info.get("partial_marks", 0),
                    "description": scoring_info.get("description", "")
                },
                "extracted_from": str(self.pdf_path.name),
                # Merge AI analysis fields directly instead of nesting
                "difficulty": analysis_result.get("difficulty", "Unknown"),
                "question_text": analysis_result.get("question_text", "")
            }

            logger.debug(f"Processed {question['subject']} Q{q_identifier}: {normalized_question_type} - Unit: {unit} - {analysis_result.get('answer', 'No answer')}")
            return question_data

        except Exception as e:
            logger.error(f"Failed to extract question {question['q_num']}: {e}")
            return None

    def _analyze_question_with_ai(self, image_path: Path, subject: str = None) -> Dict:
        """Analyze question image with AI to extract structured data"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Prepare syllabus information for the prompt
            syllabus_text = ""
            if self.syllabus:
                syllabus_text = "\n\nJEE SYLLABUS UNITS:\n"
                for subj, units in self.syllabus.items():
                    syllabus_text += f"\n{subj}:\n"
                    for unit_key, unit_name in units.items():
                        syllabus_text += f"  - {unit_name}\n"

                syllabus_text += "\nYou MUST select the unit from the above syllabus based on the question content."

            # AI prompt for analysis
            prompt = f"""
Analyze this JEE Advanced question image and extract the following information in JSON format:

{{
  "type": "MCQ" or "Numerical" or "Multiple Correct",
  "subject": "Mathematics" or "Physics" or "Chemistry",
  "unit": "EXACT unit name from the syllabus below",
  "choices": ["A", "B", "C", "D"] or [] for numerical,
  "answer": "correct answer(s) - letter(s) for MCQ or number for numerical",
  "difficulty": "Easy" or "Medium" or "Hard",
  "question_text": "EXACT question text from the image"
}}

{syllabus_text}

IMPORTANT INSTRUCTIONS:
1. For the "unit" field, you MUST use the EXACT unit name from the syllabus above
2. Analyze the question content to determine which syllabus unit it belongs to
3. Do NOT create your own unit names - only use the ones listed in the syllabus
4. Look for multiple choice options (A), (B), (C), (D)
5. Look for answer key in solution sections
6. ONLY provide valid JSON, no other text.
"""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.0,
                messages=[{
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
                            "text": prompt
                        }
                    ]
                }]
            )

            # Parse AI response
            response_text = response.content[0].text.strip()

            # Extract JSON from response
            try:
                # Look for JSON in the response
                if "{" in response_text and "}" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    json_str = response_text[json_start:json_end]
                    analysis = json.loads(json_str)

                    # Validate that the unit is from the syllabus
                    unit = analysis.get("unit", "")
                    subject_from_analysis = analysis.get("subject", "")

                    if self.syllabus and subject_from_analysis in self.syllabus:
                        valid_units = list(self.syllabus[subject_from_analysis].values())
                        if unit not in valid_units:
                            logger.warning(f"AI returned invalid unit '{unit}' for {subject_from_analysis}. Setting to 'Unknown'")
                            analysis["unit"] = "Unknown"
                        else:
                            logger.debug(f"AI selected valid unit: {unit}")

                    logger.debug(f"AI analysis successful: {analysis.get('type', 'Unknown')} question")
                    return analysis
                else:
                    logger.warning("No JSON found in AI response")
                    return {}

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AI response as JSON: {e}")
                return {}

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {}

    def save_metadata(self, questions_data: List[Dict]) -> Path:
        """Save extracted metadata to JSON file"""
        metadata_file = self.output_folder / "question_metadata_2025_advanced.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(questions_data, f, indent=2)

            # Print summary
            subjects_count = defaultdict(int)
            answers_found = 0
            question_types = defaultdict(int)

            for q in questions_data:
                subjects_count[q["subject"]] += 1
                if q.get("answer_key"):
                    answers_found += 1
                question_types[q.get("question_type", "Unknown")] += 1

            logger.info(f"Saved metadata for {len(questions_data)} questions to {metadata_file}")
            logger.info("Extraction summary:")
            for subject, count in subjects_count.items():
                logger.info(f"  {subject}: {count} questions")
            logger.info(f"  Answers found: {answers_found}/{len(questions_data)}")
            logger.info(f"  Question types: {dict(question_types)}")

            return metadata_file

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise


def extract_2025_papers_advanced():
    """Extract questions from 2025 papers using advanced method"""
    # Get list of 2025 papers
    paper_files = list(Path("data/inputs/question_papers").rglob("*2025*.pdf"))

    if not paper_files:
        logger.error("No 2025 papers found in data/inputs/question_papers/")
        return

    logger.info(f"Found {len(paper_files)} 2025 papers: {[p.name for p in paper_files]}")

    all_questions = []

    for paper_file in paper_files:
        logger.info(f"Processing {paper_file.name} with advanced extraction...")

        # Create output folder for this paper
        paper_name = paper_file.stem.replace(" ", "_")
        output_folder = f"data/outputs/extracted_questions/{paper_name}"

        try:
            extractor = AdvancedJEE2025Extractor(
                pdf_path=str(paper_file),
                output_folder=output_folder,
                syllabus_path="data/inputs/syllabus/jee_syllabus.json",
                scoring_path="data/inputs/scoring/jee_2025_scoring.json"
            )

            questions_data = extractor.extract_questions_and_answers()
            metadata_file = extractor.save_metadata(questions_data)

            all_questions.extend(questions_data)
            logger.info(f"Successfully processed {paper_file.name}")

        except Exception as e:
            logger.error(f"Failed to process {paper_file.name}: {e}")
            continue

    # Remove duplicates and save combined metadata
    if all_questions:
        # Deduplicate based on subject + question_number + source_paper
        seen_questions = set()
        unique_questions = []

        for question in all_questions:
            # Create unique identifier
            identifier = (
                question["subject"],
                question["question_number"],
                question["extracted_from"]
            )

            if identifier not in seen_questions:
                seen_questions.add(identifier)
                unique_questions.append(question)
            else:
                logger.debug(f"Removing duplicate: {question['subject']} Q{question['question_number']} from {question['extracted_from']}")

        combined_file = Path("data/outputs/extracted_questions/question_metadata_jee_2025.json")
        with open(combined_file, 'w') as f:
            json.dump(unique_questions, f, indent=2)

        logger.info(f"Saved combined metadata for {len(unique_questions)} unique questions to {combined_file}")
        logger.info(f"Removed {len(all_questions) - len(unique_questions)} duplicate questions")

        # Print summary by paper
        paper_counts = defaultdict(int)
        for q in unique_questions:
            paper_counts[q["extracted_from"]] += 1

        for paper, count in paper_counts.items():
            logger.info(f"  {paper}: {count} questions")


if __name__ == "__main__":
    extract_2025_papers_advanced()