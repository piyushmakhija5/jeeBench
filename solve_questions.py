"""
Improved question solving module with parallel processing, better error handling,
and configuration management.
"""
import os
import base64
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm

from config import config
from logger import get_logger, setup_logging
from exceptions import (
    APIKeyError, APIResponseError, APIRateLimitError, ModelNotAvailableError,
    ImageProcessingError, JSONProcessingError, ValidationError, PathNotFoundError
)

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(**config.logging)
logger = get_logger(__name__)


class QuestionProcessor:
    """Main class for processing questions with multiple AI providers"""

    def __init__(self, provider: str = "anthropic", model_name: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.model_config = config.get_model_config(self.provider, self.model_name)
        self.client = self._initialize_client()
        logger.info(f"Initialized QuestionProcessor for provider: {self.provider}, model: {self.model_config.model}")

    def _initialize_client(self):
        """Initialize the appropriate client for the provider"""
        api_key = os.getenv(self.model_config.api_key_env)
        if not api_key:
            raise APIKeyError(f"API key not found: {self.model_config.api_key_env}")

        if self.provider == "anthropic":
            return Anthropic(api_key=api_key)
        elif self.provider == "openai":
            return OpenAI(api_key=api_key)
        elif self.provider == "google":
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model_config.model)
        else:
            raise ModelNotAvailableError(f"Unsupported provider: {self.provider}")

    def encode_image(self, image_path: Union[str, Path]) -> tuple[str, float, int]:
        """Convert image to base64 string with metrics"""
        try:
            start_time = time.time()
            image_path = Path(image_path)

            if not image_path.exists():
                raise PathNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')

            encoding_time = time.time() - start_time
            encoded_size = len(encoded)

            logger.debug(f"Encoded image {image_path} in {encoding_time:.3f}s, size: {encoded_size}")
            return encoded, encoding_time, encoded_size

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise ImageProcessingError(f"Error encoding image {image_path}: {str(e)}")

    def _make_api_call(self, base64_image: str, question_data: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
        """Make API call to the provider"""
        # Get question type and generate appropriate prompt
        question_type = question_data.get("question_type", "MCQ") if question_data else "MCQ"
        prompt = self._generate_question_type_prompt(question_type, question_data)

        api_call_start = time.time()

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config.model,
                    max_tokens=self.model_config.max_tokens,
                    temperature=self.model_config.temperature,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                usage = response.usage
                response_text = response.content[0].text

            elif self.provider == "openai":
                # Prepare parameters for OpenAI API call
                api_params = {
                    "model": self.model_config.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    "temperature": self.model_config.temperature
                }

                # Use appropriate token parameter based on model
                model_name = self.model_config.model.lower()
                if any(model in model_name for model in ['o3', 'o4']):
                    api_params["max_completion_tokens"] = self.model_config.max_tokens
                else:
                    api_params["max_tokens"] = self.model_config.max_tokens

                response = self.client.chat.completions.create(**api_params)
                usage = response.usage
                response_text = response.choices[0].message.content

            elif self.provider == "google":
                response = self.client.generate_content(
                    contents=[{
                        "parts": [
                            {"mime_type": "image/png", "data": base64.b64decode(base64_image)},
                            {"text": prompt}
                        ]
                    }],
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.model_config.temperature,
                        candidate_count=1,
                        max_output_tokens=self.model_config.max_tokens
                    )
                )
                # print(response)
                # Extract actual usage metadata from Gemini response
                usage_metadata = getattr(response, 'usage_metadata', None)
                if usage_metadata:
                    # Create usage object with Gemini's actual token counts
                    usage = usage_metadata
                else:
                    # Fallback to fake usage object if no usage metadata
                    usage = type('Usage', (), {
                        "prompt_token_count": 0,
                        "candidates_token_count": 0,
                        "total_token_count": 0
                    })()
                response_text = response.text

            api_call_time = time.time() - api_call_start

            # Calculate costs - handle different provider token usage formats
            if self.provider == "anthropic":
                # Anthropic uses input_tokens and output_tokens
                input_tokens = getattr(usage, 'input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0)
            elif self.provider == "openai":
                # OpenAI uses prompt_tokens and completion_tokens
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
            elif self.provider == "google":
                # Google/Gemini uses prompt_token_count and candidates_token_count
                # Try both snake_case and camelCase variations
                input_tokens = (getattr(usage, 'prompt_token_count', 0) or
                               getattr(usage, 'promptTokenCount', 0) or
                               getattr(usage, 'input_tokens', 0) or
                               getattr(usage, 'prompt_tokens', 0))
                output_tokens = (getattr(usage, 'candidates_token_count', 0) or
                                getattr(usage, 'candidatesTokenCount', 0) or
                                getattr(usage, 'output_tokens', 0) or
                                getattr(usage, 'completion_tokens', 0))
            else:
                # Default fallback - try multiple naming conventions
                input_tokens = (getattr(usage, 'input_tokens', 0) or
                               getattr(usage, 'prompt_tokens', 0) or
                               getattr(usage, 'prompt_token_count', 0) or
                               getattr(usage, 'promptTokenCount', 0))
                output_tokens = (getattr(usage, 'output_tokens', 0) or
                                getattr(usage, 'completion_tokens', 0) or
                                getattr(usage, 'candidates_token_count', 0) or
                                getattr(usage, 'candidatesTokenCount', 0))

            input_cost = input_tokens * self.model_config.input_price
            output_cost = output_tokens * self.model_config.output_price

            metrics = {
                "api_call_time": api_call_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }

            logger.debug(f"API call completed in {api_call_time:.3f}s, tokens: {input_tokens + output_tokens}")
            return response_text, metrics

        except Exception as e:
            logger.error(f"API call failed: {e}")
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise APIResponseError(f"API call failed: {e}")

    def _generate_question_type_prompt(self, question_type: str, question_data: Dict[str, Any] = None) -> str:
        """Generate question-type specific prompt"""
        base_instruction = "Analyze this question and provide your response in the following JSON format:\n"

        if question_type == "MCQ" or question_type == "Pair Matching":
            choices = question_data.get("choices", ["A", "B", "C", "D"]) if question_data else ["A", "B", "C", "D"]
            choices_str = ", ".join(choices)

            prompt = (
                base_instruction +
                "{\n"
                '  "Solution": "Detailed step-by-step solution explanation",\n'
                f'  "Final Answer": "Single letter from [{choices_str}] - ONLY the letter, no additional text"\n'
                "}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                f"- Final Answer MUST be exactly ONE letter: {choices_str}\n"
                "- Do NOT include parentheses, explanations, or additional text\n"
                "- Examples of CORRECT format: 'A', 'B', 'C', 'D'\n"
                f"- Examples of INCORRECT format: '(A)', 'Choice A', 'A is correct', 'Answer is A'\n"
                "- You MUST select one of the given choices even if uncertain\n"
                "ONLY provide valid JSON structure and escape any special characters properly."
            )

        elif question_type == "Multiple Correct":
            choices = question_data.get("choices", ["A", "B", "C", "D"]) if question_data else ["A", "B", "C", "D"]
            choices_str = ", ".join(choices)

            prompt = (
                base_instruction +
                "{\n"
                '  "Solution": "Detailed step-by-step solution explanation",\n'
                f'  "Final Answer": "One or more letters from [{choices_str}] separated by commas - ONLY the letters"\n'
                "}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                f"- Final Answer MUST be one or more letters from: {choices_str}\n"
                "- Multiple answers should be separated by commas: 'A, B' or 'A, C, D'\n"
                "- Do NOT include parentheses, explanations, or additional text\n"
                "- Examples of CORRECT format: 'A', 'A, B', 'A, C, D', 'B, D'\n"
                "- Examples of INCORRECT format: '(A), (B)', 'Choices A and B', 'A and B are correct'\n"
                "- You MUST select at least one choice\n"
                "ONLY provide valid JSON structure and escape any special characters properly."
            )

        elif question_type == "Numerical":
            prompt = (
                base_instruction +
                "{\n"
                '  "Solution": "Detailed step-by-step solution explanation",\n'
                '  "Final Answer": "Numerical value only - no units, no additional text"\n'
                "}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "- Final Answer MUST be a numerical value only\n"
                "- Do NOT include units, explanations, or additional text\n"
                "- Use appropriate precision (typically 2-3 decimal places)\n"
                "- Examples of CORRECT format: '5', '3.14', '0.75', '-2.5'\n"
                "- Examples of INCORRECT format: '5 meters', 'approximately 3.14', 'Answer is 0.75'\n"
                "ONLY provide valid JSON structure and escape any special characters properly."
            )

        else:
            # Default fallback
            prompt = (
                base_instruction +
                "{\n"
                '  "Solution": "Detailed step-by-step solution explanation",\n'
                '  "Final Answer": "MCQ answer choice(s) or the final numerical answer"\n'
                "}\n"
                "ONLY provide valid JSON structure and escape any special characters properly."
            )

        return prompt

    def process_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single question and return results"""
        try:
            # Validate input
            if not isinstance(question_data, dict) or "image_path" not in question_data:
                raise ValidationError(f"Invalid question format: {question_data}")

            image_path = question_data["image_path"]
            logger.info(f"Processing question: {image_path}")

            # Encode image
            base64_image, encoding_time, encoded_size = self.encode_image(image_path)

            # Make API call with retry logic
            response_text, api_metrics = self._make_api_call_with_retry(base64_image, question_data)

            # Parse response
            final_answer = self._extract_final_answer(response_text, question_data)

            # Get answer key - handle both string and list formats
            answer_key = question_data.get("answer_key")

            # Get scoring information from metadata and add question_type
            scoring_info = question_data.get("scoring", {}).copy()
            if 'question_type' not in scoring_info:
                scoring_info['question_type'] = question_data.get("question_type")

            # Calculate match and score with proper JEE scoring
            match_type, score = self._compare_answers_with_score(final_answer, answer_key, scoring_info)

            # Combine metrics
            metrics = {
                "encoding_time": encoding_time,
                "encoded_size": encoded_size,
                **api_metrics
            }

            result = {
                "metadata": question_data,
                "processing_result": {
                    "metrics": metrics,
                    "response": response_text
                },
                "final_answer": final_answer,
                "match": match_type,
                "score": score
            }

            logger.info(f"Successfully processed {image_path}, match: {result['match']}, score: {score}")
            return result

        except Exception as e:
            logger.error(f"Failed to process question {question_data.get('image_path', 'unknown')}: {e}")
            return {
                "metadata": question_data,
                "processing_result": {
                    "error": str(e),
                    "metrics": None,
                    "response": None
                },
                "final_answer": None,
                "match": "Error",
                "score": 0.0
            }

    def _make_api_call_with_retry(self, base64_image: str, question_data: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
        """Make API call with retry logic"""
        for attempt in range(config.processing.retry_attempts):
            try:
                return self._make_api_call(base64_image, question_data)
            except APIRateLimitError as e:
                if attempt < config.processing.retry_attempts - 1:
                    wait_time = config.processing.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                if attempt < config.processing.retry_attempts - 1:
                    logger.warning(f"API call failed, retrying... (attempt {attempt + 1}): {e}")
                    time.sleep(config.processing.retry_delay)
                else:
                    raise e

    def _extract_final_answer(self, response_text: str, question_data: Dict[str, Any] = None) -> Optional[str]:
        """Extract the final answer from the LLM response"""
        if not response_text:
            return None

        try:
            # Strategy 1: Try JSON parsing (handles most cases)
            answer = self._try_json_extraction(response_text)
            if answer:
                # Clean and validate the answer based on question type
                cleaned_answer = self._clean_and_validate_answer(answer, question_data)
                return cleaned_answer

            # Strategy 2: Regex fallback for "Final Answer" patterns
            answer = self._try_regex_extraction(response_text)
            if answer:
                # Clean and validate the answer based on question type
                cleaned_answer = self._clean_and_validate_answer(answer, question_data)
                return cleaned_answer

            return None

        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return None

    def _clean_and_validate_answer(self, answer: str, question_data: Dict[str, Any] = None) -> str:
        """Clean and validate answer based on question type"""
        if not answer:
            return answer

        question_type = question_data.get("question_type", "MCQ") if question_data else "MCQ"
        choices = question_data.get("choices", ["A", "B", "C", "D"]) if question_data else ["A", "B", "C", "D"]

        # Clean common unwanted patterns
        import re
        cleaned_answer = answer.strip()

        if question_type == "MCQ" or question_type == "Pair Matching":
            # Extract single letter choice
            # Remove common prefixes and suffixes
            cleaned_answer = re.sub(r'^(Answer\s*is\s*|Choice\s*|Option\s*|\(|\))', '', cleaned_answer, flags=re.IGNORECASE)
            cleaned_answer = re.sub(r'(\s*is\s*correct|\s*is\s*the\s*answer|\)|\.)$', '', cleaned_answer, flags=re.IGNORECASE)

            # Extract the first valid choice letter
            for choice in choices:
                if re.search(rf'\b{choice}\b', cleaned_answer, re.IGNORECASE):
                    return choice.upper()

            # If no valid choice found, try to extract any letter
            letter_match = re.search(r'\b([ABCD])\b', cleaned_answer, re.IGNORECASE)
            if letter_match:
                letter = letter_match.group(1).upper()
                if letter in choices:
                    return letter

        elif question_type == "Multiple Correct":
            # Extract multiple choice letters
            # Clean common patterns
            cleaned_answer = re.sub(r'(Answer\s*is\s*|Choices?\s*|Options?\s*)', '', cleaned_answer, flags=re.IGNORECASE)
            cleaned_answer = re.sub(r'(\s*are\s*correct|\s*is\s*the\s*answer)$', '', cleaned_answer, flags=re.IGNORECASE)

            # Find all valid choice letters
            found_choices = []
            for choice in choices:
                if re.search(rf'\b{choice}\b', cleaned_answer, re.IGNORECASE):
                    if choice.upper() not in found_choices:
                        found_choices.append(choice.upper())

            if found_choices:
                # Sort choices to maintain consistent order
                found_choices.sort()
                return ", ".join(found_choices)

        elif question_type == "Numerical":
            # Extract numerical value
            # Remove units and explanatory text
            cleaned_answer = re.sub(r'(approximately|about|roughly|\u2248|~)', '', cleaned_answer, flags=re.IGNORECASE)

            # Extract the first number (including decimals and negative numbers)
            number_match = re.search(r'-?\d+\.?\d*', cleaned_answer)
            if number_match:
                return number_match.group(0)

        # Return original if no specific cleaning was applied
        return cleaned_answer

    def _try_json_extraction(self, response_text: str) -> Optional[str]:
        """Try to extract answer from JSON"""
        import re

        # Clean up common JSON wrapper patterns
        text = response_text.strip()

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Find JSON-like content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None

        try:
            json_content = json_match.group(0)
            parsed = json.loads(json_content)

            # Try common key variations
            for key in ["Final Answer", "final answer", "Final_Answer", "answer"]:
                if key in parsed:
                    answer = parsed[key]
                    if answer and str(answer).strip():
                        return str(answer).strip()

            return None

        except json.JSONDecodeError:
            return None

    def _try_regex_extraction(self, response_text: str) -> Optional[str]:
        """Extract using regex patterns as fallback"""
        import re

        # Look for "Final Answer:" patterns, including bold markdown
        patterns = [
            r'"Final Answer":\s*"([^"]+)"',
            r'"Final Answer":\s*([^,}\n]+)',
            r'\*\*Final Answer:\*\*\s*([^\n]+)',  # Handle bold markdown
            # Add patterns for LaTeX \boxed{} format BEFORE the generic patterns
            r'\\boxed\{([^}]+)\}',  # Match \boxed{content}
            r'\$\\boxed\{([^}]+)\}\$',  # Match $\boxed{content}$
            r'is\s+\$\\boxed\{([^}]+)\}\$',  # Match "is $\boxed{content}$"
            # Generic patterns should come AFTER LaTeX patterns
            r'Final Answer:?\s*([^\n]+)',
            r'Answer:?\s*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts and bold markdown
                answer = re.sub(r'^["\',;:*]+|["\',;:*]+$', '', answer)
                if answer:
                    return answer

        return None

    def _compare_answers_with_score(self, llm_answer: Optional[str], key_answer: Optional[str], scoring_info: Dict[str, Any] = None) -> Tuple[str, float]:
        """Compare LLM answer with answer key and return match type and score"""
        try:
            if llm_answer is None or key_answer is None:
                return "No", 0.0

            # Get question type from scoring_info if available
            question_type = None
            if scoring_info:
                # Check if scoring_info contains question_type directly or if we need to get it from description
                question_type = scoring_info.get('question_type')
                if not question_type:
                    # Try to infer from description
                    description = scoring_info.get('description', '')
                    if 'Multiple Correct' in description:
                        question_type = 'Multiple Correct'
                    elif 'Pair Matching' in description:
                        question_type = 'Pair Matching'
                    elif 'Single Correct' in description or 'MCQ' in description:
                        question_type = 'MCQ'
                    elif 'Numerical' in description:
                        question_type = 'Numerical'

            # ONLY Multiple Correct questions can have multiple answers and partial marks
            if question_type == "Multiple Correct":
                # Always treat as multiple correct if question type is Multiple Correct
                if isinstance(key_answer, list):
                    return self._compare_multiple_correct_answers_with_score(llm_answer, key_answer, scoring_info)
                elif isinstance(key_answer, str) and ',' in key_answer:
                    # Convert comma-separated string to list
                    key_answers_list = [choice.strip() for choice in key_answer.split(',')]
                    return self._compare_multiple_correct_answers_with_score(llm_answer, key_answers_list, scoring_info)
                else:
                    # Single answer key but Multiple Correct question type - convert to list
                    key_answers_list = [key_answer.strip()]
                    return self._compare_multiple_correct_answers_with_score(llm_answer, key_answers_list, scoring_info)

            # For all other question types (MCQ, Numerical, Pair Matching), handle as single answer
            # Even if key_answer is a list or comma-separated, these question types don't have partial scoring
            elif isinstance(key_answer, list):
                # If it's a list but not Multiple Correct, check if LLM answer matches any of the valid answers
                for single_key in key_answer:
                    match_type, score = self._compare_single_answer_with_score(llm_answer, single_key, scoring_info)
                    if match_type == "Yes":
                        return match_type, score
                # No match found
                return "No", scoring_info.get('negative_marks', 0) if scoring_info else 0.0
            elif isinstance(key_answer, str) and ',' in key_answer:
                # If it's comma-separated but not Multiple Correct, check each possibility
                key_answers_list = [choice.strip() for choice in key_answer.split(',')]
                for single_key in key_answers_list:
                    match_type, score = self._compare_single_answer_with_score(llm_answer, single_key, scoring_info)
                    if match_type == "Yes":
                        return match_type, score
                # No match found
                return "No", scoring_info.get('negative_marks', 0) if scoring_info else 0.0

            # Handle single answer (most common case)
            return self._compare_single_answer_with_score(llm_answer, key_answer, scoring_info)

        except Exception as e:
            logger.error(f"Error comparing answers: {e}")
            return "No", 0.0

    def _compare_single_answer_with_score(self, llm_answer: str, key_answer: str, scoring_info: Dict[str, Any] = None) -> Tuple[str, float]:
        """Compare single answer responses with scoring - NO partial marks for single answer questions"""
        # Get scoring information from the question metadata
        if scoring_info is None:
            scoring_info = {}

        # Only use scoring values if explicitly provided in metadata
        full_marks = scoring_info.get('full_marks')
        negative_marks = scoring_info.get('negative_marks')
        zero_marks = scoring_info.get('zero_marks', 0)

        # Normalize both answers
        llm_norm = self._normalize_answer(llm_answer)
        key_norm = self._normalize_answer(key_answer)

        # Direct match - award full marks (or use 0 if no scoring info)
        if llm_norm == key_norm or self._is_numerical_match(llm_answer, key_answer):
            return "Yes", full_marks if full_marks is not None else 0.0

        # For single answer questions (MCQ, Numerical, Pair Matching), there are NO partial marks
        # Only full marks for correct, negative/zero marks for incorrect, zero marks for unanswered

        # Incorrect answer - award negative marks (or zero for Numerical questions)
        return "No", negative_marks if negative_marks is not None else 0.0

    def _compare_answers(self, llm_answer: Optional[str], key_answer: Optional[str]) -> bool:
        """Compare LLM answer with answer key (backward compatibility)"""
        match_type, _ = self._compare_answers_with_score(llm_answer, key_answer)
        return match_type == "Yes"

    def _compare_single_answer(self, llm_answer: str, key_answer: str) -> bool:
        """Compare single answer responses (backward compatibility)"""
        match_type, _ = self._compare_single_answer_with_score(llm_answer, key_answer)
        return match_type == "Yes"

    def _compare_multiple_correct_answers_with_score(self, llm_answer: str, key_answers: List[str], scoring_info: Dict[str, Any] = None) -> Tuple[str, float]:
        """Compare multiple correct answer responses with JEE 2025 scoring logic"""
        import re

        # Get scoring information from the question metadata
        if scoring_info is None:
            scoring_info = {}

        # JEE Multi-correct standard scores (use provided values or JEE defaults)
        full_marks = scoring_info.get('full_marks', 4)
        negative_marks = scoring_info.get('negative_marks', -2)
        zero_marks = scoring_info.get('zero_marks', 0)

        # JEE partial marks scheme (use provided values or JEE defaults)
        partial_marks_config = scoring_info.get('partial_marks', {})
        partial_3_of_4 = partial_marks_config.get('3_of_4', 3)
        partial_2_of_3_plus = partial_marks_config.get('2_of_3_plus', 2)
        partial_1_of_2_plus = partial_marks_config.get('1_of_2_plus', 1)

        # Extract individual choices from LLM answer
        llm_choices = self._extract_choices_from_response(llm_answer)

        # Normalize key answers
        key_choices = set(self._normalize_answer(choice) for choice in key_answers)

        # Handle unanswered case - no choices selected
        if not llm_choices:
            return "No", zero_marks

        # Handle edge case - no correct answers exist (shouldn't happen in real JEE)
        if not key_choices:
            return "No", negative_marks

        # Calculate overlap and incorrect selections
        overlap = llm_choices.intersection(key_choices)
        incorrect_choices = llm_choices - key_choices

        total_correct = len(key_choices)
        selected_correct = len(overlap)
        selected_incorrect = len(incorrect_choices)

        # JEE Rule: ANY incorrect choice selected = -2 marks (no exceptions)
        # "Negative Marks: −2 In all other cases"
        if selected_incorrect > 0:
            return "No", negative_marks

        # Only correct choices selected (no incorrect ones)
        if selected_correct > 0:
            # Full match - all correct choices selected
            # "Full Marks: +4 ONLY if (all) the correct option(s) is(are) chosen"
            if llm_choices == key_choices:
                return "Yes", full_marks

            # Partial credit cases (JEE 2025 specific rules)
            # These rules only apply when NO incorrect choices are selected

            # Rule: "+3 If all the four options are correct but ONLY three options are chosen"
            if total_correct == 4 and selected_correct == 3:
                return "Partial", partial_3_of_4

            # Rule: "+2 If three or more options are correct but ONLY two options are chosen, both correct"
            elif total_correct >= 3 and selected_correct == 2:
                return "Partial", partial_2_of_3_plus

            # Rule: "+1 If two or more options are correct but ONLY one option is chosen and it is correct"
            elif total_correct >= 2 and selected_correct == 1:
                return "Partial", partial_1_of_2_plus

            # Edge case: If there's only 1 correct answer and it's selected
            # This should be full marks since ALL correct answers are chosen
            elif total_correct == 1 and selected_correct == 1:
                return "Yes", full_marks

            # Any other partial selection that doesn't match JEE rules = negative marks
            else:
                return "No", negative_marks

        # No correct choices selected = negative marks
        return "No", negative_marks

    def _compare_multiple_correct_answers(self, llm_answer: str, key_answers: List[str]) -> bool:
        """Compare multiple correct answer responses (backward compatibility)"""
        match_type, _ = self._compare_multiple_correct_answers_with_score(llm_answer, key_answers)
        return match_type == "Yes"

    def _is_numerical_match(self, llm_answer: str, key_answer: str) -> bool:
        """Check if numerical answers match within tolerance or handle OR conditions"""
        import re

        try:
            # FIX: Handle OR conditions in numerical answers first
            if ' OR ' in key_answer.upper():
                # Split by OR and check if LLM answer matches any of the alternatives
                or_parts = re.split(r'\s+OR\s+', key_answer, flags=re.IGNORECASE)
                for part in or_parts:
                    part = part.strip()
                    # Check direct match or numerical match for each part
                    if self._normalize_answer(llm_answer) == self._normalize_answer(part):
                        return True
                    # Also check numerical tolerance for each part
                    if self._check_numerical_tolerance(llm_answer, part):
                        return True
                return False

            # Check if answer key is a range format
            if self._is_range_answer(key_answer):
                return self._is_numerical_in_range(llm_answer, key_answer)

            # For non-range answers, use existing numerical comparison
            return self._check_numerical_tolerance(llm_answer, key_answer)

        except (ValueError, IndexError):
            return False

    def _check_numerical_tolerance(self, llm_answer: str, key_answer: str) -> bool:
        """Check if two numerical answers match within tolerance"""
        import re

        try:
            # Extract numbers from both answers
            llm_numbers = re.findall(r'-?\d+\.?\d*', llm_answer)
            key_numbers = re.findall(r'-?\d+\.?\d*', key_answer)

            if not llm_numbers or not key_numbers:
                return False

            # Convert to float and compare with tolerance
            llm_val = float(llm_numbers[0])
            key_val = float(key_numbers[0])

            # Use relative tolerance of 1% or absolute tolerance of 0.01
            tolerance = max(0.01, abs(key_val) * 0.01)
            return abs(llm_val - key_val) <= tolerance

        except (ValueError, IndexError):
            return False

    def _is_range_answer(self, answer: str) -> bool:
        """Check if answer is in range format like '[0.7 to 0.8]' or '0.27 to 0.33'"""
        import re

        # FIX: Don't treat OR conditions as ranges - they are separate valid answers
        if ' OR ' in answer.upper():
            return False

        # Check for range patterns and negative numbers
        range_patterns = [
            r'\[[\-]?[\d.]+\s+to\s+[\-]?[\d.]+\]',  # [0.7 to 0.8] or [-7.2 to -7]
            r'[\-]?[\d.]+\s+to\s+[\-]?[\d.]+',       # 0.27 to 0.33 or -7.2 to -7
            r'\[[\-]?[\d.]+\s*-\s*[\-]?[\d.]+\]',    # [0.7-0.8] with negative support
            r'[\-]?[\d.]+\s*-\s*[\-]?[\d.]+',        # 0.7-0.8 with negative support
        ]

        answer_cleaned = answer.strip()
        for pattern in range_patterns:
            if re.search(pattern, answer_cleaned, re.IGNORECASE):
                return True
        return False

    def _is_numerical_in_range(self, llm_answer: str, range_answer: str) -> bool:
        """Check if numerical answer lies within the specified range"""
        import re

        try:
            # Extract the numerical value from LLM answer
            llm_numbers = re.findall(r'-?\d+\.?\d*', llm_answer)
            if not llm_numbers:
                return False

            llm_val = float(llm_numbers[0])

            # Extract range bounds from the answer key
            # Handle different range formats
            range_bounds = self._extract_range_bounds(range_answer)
            if not range_bounds:
                return False

            min_val, max_val = range_bounds

            # Check if the value lies within the range (inclusive)
            return min_val <= llm_val <= max_val

        except (ValueError, IndexError):
            return False

    def _extract_range_bounds(self, range_answer: str) -> Optional[Tuple[float, float]]:
        """Extract minimum and maximum values from range answer"""
        import re

        try:
            # Remove brackets and normalize
            cleaned = range_answer.strip().replace('[', '').replace(']', '')

            # Handle OR cases (e.g., "-29.95 to -29.8 OR 29.8 to 29.95")
            if ' OR ' in cleaned.upper():
                or_parts = re.split(r'\s+OR\s+', cleaned, flags=re.IGNORECASE)
                # Try each OR part and return the first valid one
                for part in or_parts:
                    bounds = self._extract_single_range_bounds(part.strip())
                    if bounds:
                        return bounds
                return None
            else:
                return self._extract_single_range_bounds(cleaned)

        except (ValueError, IndexError):
            return None

    def _extract_single_range_bounds(self, range_str: str) -> Optional[Tuple[float, float]]:
        """Extract bounds from a single range string"""
        import re

        try:
            # Handle different separators with improved negative number support
            if ' to ' in range_str:
                parts = range_str.split(' to ')
            elif ' TO ' in range_str:
                parts = range_str.split(' TO ')
            else:
                # Handle dash separator while preserving negative signs
                # Look for pattern: number - number (not negative number)
                # Use word boundaries and spacing to distinguish
                dash_pattern = r'(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)'
                match = re.match(dash_pattern, range_str.strip())
                if match:
                    parts = [match.group(1), match.group(2)]
                else:
                    return None

            if len(parts) != 2:
                return None

            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())

            # Ensure min_val <= max_val
            if min_val > max_val:
                min_val, max_val = max_val, min_val

            return (min_val, max_val)

        except (ValueError, IndexError):
            return None

    def _extract_choices_from_response(self, response: str) -> set:
        """Extract choice letters (A, B, C, D) from LLM response"""
        import re

        if not response:
            return set()

        # Normalize the response
        response = response.upper()

        # Pattern to find choice letters
        # Matches patterns like "(A)", "A", "(A) and (B)", "A, B", etc.
        choice_patterns = [
            r'\b([ABCD])\b',  # Single letters
            r'\(([ABCD])\)',  # Letters in parentheses
        ]

        choices = set()
        for pattern in choice_patterns:
            matches = re.findall(pattern, response)
            choices.update(matches)

        # Additional logic to handle common response formats
        # Handle "and" connections: "A and B", "(A) and (C)"
        and_pattern = r'(?:\(([ABCD])\)|([ABCD]))\s+and\s+(?:\(([ABCD])\)|([ABCD]))'
        and_matches = re.findall(and_pattern, response)
        for match_group in and_matches:
            for choice in match_group:
                if choice:
                    choices.add(choice)

        # Handle comma-separated: "A, B", "(A), (C)"
        comma_pattern = r'(?:\(([ABCD])\)|([ABCD]))\s*,\s*(?:\(([ABCD])\)|([ABCD]))'
        comma_matches = re.findall(comma_pattern, response)
        for match_group in comma_matches:
            for choice in match_group:
                if choice:
                    choices.add(choice)

        return choices

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison"""
        if answer is None:
            return ""
        # Convert to string and uppercase
        answer = str(answer).upper()
        # Remove common decorators
        answer = answer.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')
        return answer


class BatchProcessor:
    """Process multiple questions in parallel"""

    def __init__(self, provider: str = "anthropic", model_name: str = None):
        self.provider = provider
        self.model_name = model_name
        self.processor = QuestionProcessor(provider, model_name)
        logger.info(f"Initialized BatchProcessor for provider: {provider}, model: {model_name}")

    def process_questions_parallel(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process questions in parallel using ThreadPoolExecutor"""
        results = []

        with ThreadPoolExecutor(max_workers=config.processing.max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self.processor.process_single_question, question): question
                for question in questions
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_question),
                             total=len(questions),
                             desc=f"Processing questions ({self.provider}/{self.model_name})"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    question = future_to_question[future]
                    logger.error(f"Failed to process question {question.get('image_path', 'unknown')}: {e}")
                    results.append({
                        "metadata": question,
                        "processing_result": {"error": str(e)},
                        "final_answer": None,
                        "match": "Error",
                        "score": 0.0
                    })

        return results


def load_questions(metadata_file: Union[str, Path] = "question_metadata.json") -> List[Dict[str, Any]]:
    """Load questions from metadata file"""
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise PathNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            questions = json.load(f)

        if not questions:
            raise ValidationError("No questions found in metadata file")

        logger.info(f"Loaded {len(questions)} questions from {metadata_path}")
        return questions

    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        raise


def save_results(results: List[Dict[str, Any]],
                provider: str,
                model: str,
                output_dir: Path = Path(".")) -> Path:
    """Save results to JSON file"""
    try:
        # Calculate summary statistics
        total_questions = len(results)
        successful_results = [r for r in results if r.get("match") != "Error"]

        # Count different match types
        yes_matches = sum(1 for r in successful_results if r.get("match") == "Yes")
        partial_matches = sum(1 for r in successful_results if r.get("match") == "Partial")
        no_matches = sum(1 for r in successful_results if r.get("match") == "No")

        # Calculate total score
        total_score = sum(r.get("score", 0) for r in successful_results)

        # Calculate token usage
        total_input_tokens = sum(
            r.get("processing_result", {}).get("metrics", {}).get("input_tokens", 0)
            for r in successful_results
        )
        total_output_tokens = sum(
            r.get("processing_result", {}).get("metrics", {}).get("output_tokens", 0)
            for r in successful_results
        )
        total_tokens = sum(
            r.get("processing_result", {}).get("metrics", {}).get("total_tokens", 0)
            for r in successful_results
        )
        total_cost = sum(
            r.get("processing_result", {}).get("metrics", {}).get("total_cost", 0)
            for r in successful_results
        )

        # Create output structure
        output_data = {
            "total_questions": total_questions,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
            "summary": {
                "successful_processing": len(successful_results),
                "yes_matches": yes_matches,
                "partial_matches": partial_matches,
                "no_matches": no_matches,
                "accuracy": yes_matches / len(successful_results) if successful_results else 0,
                "total_score": round(total_score, 2),
                "score_percentage": round((total_score / 360 * 100), 2),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost
            },
            "questions": results
        }

        # Save to file
        output_file = output_dir / f"jee_questions_{provider}_{model.replace('/', '_')}_response.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Results summary:")
        logger.info(f"  - Yes matches: {yes_matches}/{len(successful_results)}")
        logger.info(f"  - Partial matches: {partial_matches}/{len(successful_results)}")
        logger.info(f"  - No matches: {no_matches}/{len(successful_results)}")
        logger.info(f"  - Accuracy: {output_data['summary']['accuracy']:.2%}")
        logger.info(f"  - Total score: {total_score:.2f} ({output_data['summary']['score_percentage']:.2f}%)")
        logger.info(f"  - Total cost: ${total_cost:.4f}")

        return output_file

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def check_output_file_exists(provider: str, model: str, output_dir: Path) -> bool:
    """Check if output file already exists"""
    output_file = output_dir / f"jee_questions_{provider}_{model.replace('/', '_')}_response.json"
    return output_file.exists()


def get_provider_model_combinations(provider: str, model: str) -> List[tuple[str, str]]:
    """Get list of (provider, model) combinations based on input parameters"""
    if provider.lower() == "all":
        return config.get_all_provider_model_combinations()
    elif model and model.lower() == "all":
        models = config.get_all_models_for_provider(provider)
        return [(provider, model_name) for model_name in models.keys()]
    elif model:
        # Validate that the specific model exists for the provider
        models = config.get_all_models_for_provider(provider)
        if model not in models:
            raise ValueError(f"Model '{model}' not found for provider '{provider}'. Available models: {list(models.keys())}")
        return [(provider, model)]
    else:
        # Default behavior - use first model for the provider
        models = config.get_all_models_for_provider(provider)
        first_model = list(models.keys())[0]
        return [(provider, first_model)]


def process_single_provider_model(provider: str, model_name: str, questions: List[Dict[str, Any]],
                                 output_dir: Path, parallel: bool) -> Optional[Path]:
    """Process questions for a single provider/model combination"""
    logger.info(f"Processing with {provider}/{model_name}")

    # Check if output file already exists
    if check_output_file_exists(provider, model_name, output_dir):
        output_file = output_dir / f"jee_questions_{provider}_{model_name.replace('/', '_')}_response.json"
        logger.info(f"Output file {output_file} already exists. Skipping processing for {provider}/{model_name}")
        return None

    try:
        # Get specific model config
        model_config = config.get_model_config(provider, model_name)

        # Process questions
        if parallel:
            processor = BatchProcessor(provider, model_name)
            results = processor.process_questions_parallel(questions)
        else:
            processor = QuestionProcessor(provider, model_name)
            results = []
            for question in tqdm(questions, desc=f"Processing questions ({provider}/{model_name})"):
                result = processor.process_single_question(question)
                results.append(result)

        # Save results
        output_file = save_results(results, provider, model_config.model, output_dir)
        logger.info(f"Completed processing for {provider}/{model_name}")
        return output_file

    except Exception as e:
        logger.error(f"Failed to process {provider}/{model_name}: {e}", exc_info=True)
        return None


def main():
    """Main function to process questions"""
    import argparse

    parser = argparse.ArgumentParser(description='Process questions using AI model providers')
    parser.add_argument('--provider', type=str, default='anthropic',
                       help='The AI model provider to use (anthropic, openai, google, or "all" for all providers)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model name to use, or "all" to run all models for the provider')
    parser.add_argument('--questions', type=str, default='data/outputs/extracted_questions/question_metadata_jee_2025.json',
                       help='Path to questions metadata file')
    parser.add_argument('--output', type=str, default='data/outputs/results',
                       help='Output directory for results')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging with specified level
    setup_logging(level=args.log_level, **{k: v for k, v in config.logging.items() if k != 'level'})

    try:
        # Validate API keys
        config.validate_api_keys()

        # Load questions
        questions = load_questions(args.questions)

        # Get provider/model combinations to process
        combinations = get_provider_model_combinations(args.provider, args.model)

        logger.info(f"Found {len(combinations)} provider/model combinations to process")
        for provider, model in combinations:
            logger.info(f"  - {provider}/{model}")

        # Process each combination
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        successful_runs = 0
        skipped_runs = 0
        failed_runs = 0

        for provider, model_name in combinations:
            output_file = process_single_provider_model(
                provider, model_name, questions, output_dir, args.parallel
            )

            if output_file:
                successful_runs += 1
            elif check_output_file_exists(provider, model_name, output_dir):
                skipped_runs += 1
            else:
                failed_runs += 1

        logger.info(f"Processing summary:")
        logger.info(f"  - Successful runs: {successful_runs}")
        logger.info(f"  - Skipped runs (files exist): {skipped_runs}")
        logger.info(f"  - Failed runs: {failed_runs}")

        if failed_runs > 0:
            logger.warning("Some processing runs failed. Check logs for details.")
            return 1

        logger.info("All processing completed successfully!")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())