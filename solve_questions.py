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
from typing import Dict, Any, List, Optional, Union
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
    
    def _make_api_call(self, base64_image: str) -> tuple[str, Dict[str, Any]]:
        """Make API call to the provider"""
        prompt = (
            "Analyze this question and provide your response in the following JSON format:\n"
            "{\n"
            '  "Solution": "Detailed step-by-step solution explanation",\n'
            '  "Final Answer": "MCQ answer choice(s) or the final numerical answer"\n'
            "}\n"
            "ONLY provide valid JSON structure and escape any special characters properly."
        )
        
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
                # Gemini doesn't provide detailed token usage
                usage = type('Usage', (), {"prompt_tokens": 0, "completion_tokens": 0})()
                response_text = response.text
            
            api_call_time = time.time() - api_call_start
            
            # Calculate costs
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
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
            response_text, api_metrics = self._make_api_call_with_retry(base64_image)
            
            # Parse response
            final_answer = self._extract_final_answer(response_text)
            
            # Get answer key - handle both string and list formats
            answer_key = question_data.get("answer_key")
            
            # Calculate match
            answer_match = self._compare_answers(final_answer, answer_key)
            
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
                "match": "Yes" if answer_match else "No"
            }
            
            logger.info(f"Successfully processed {image_path}, match: {result['match']}")
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
                "match": "Error"
            }
    
    def _make_api_call_with_retry(self, base64_image: str) -> tuple[str, Dict[str, Any]]:
        """Make API call with retry logic"""
        for attempt in range(config.processing.retry_attempts):
            try:
                return self._make_api_call(base64_image)
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
    
    def _extract_final_answer(self, response_text: str) -> Optional[str]:
        """Extract the final answer from the LLM response"""
        if not response_text:
            return None
        
        try:
            # Strategy 1: Try JSON parsing (handles most cases)
            answer = self._try_json_extraction(response_text)
            if answer:
                return answer
            
            # Strategy 2: Regex fallback for "Final Answer" patterns
            answer = self._try_regex_extraction(response_text)
            if answer:
                return answer
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return None
    
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
        
        # Look for "Final Answer:" patterns
        patterns = [
            r'"Final Answer":\s*"([^"]+)"',
            r'"Final Answer":\s*([^,}\n]+)',
            r'Final Answer:?\s*([^\n]+)',
            r'Answer:?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts
                answer = re.sub(r'^["\',;:]+|["\',;:]+$', '', answer)
                if answer:
                    return answer
        
        return None
    
    def _compare_answers(self, llm_answer: Optional[str], key_answer: Optional[str]) -> bool:
        """Compare LLM answer with answer key"""
        try:
            if llm_answer is None or key_answer is None:
                return False
            
            # Handle multiple correct answers (when key_answer is a list)
            if isinstance(key_answer, list):
                return self._compare_multiple_correct_answers(llm_answer, key_answer)
            
            # Handle single answer (backward compatibility)
            return self._compare_single_answer(llm_answer, key_answer)
            
        except Exception as e:
            logger.error(f"Error comparing answers: {e}")
            return False
    
    def _compare_single_answer(self, llm_answer: str, key_answer: str) -> bool:
        """Compare single answer responses"""
        # Normalize both answers
        llm_norm = self._normalize_answer(llm_answer)
        key_norm = self._normalize_answer(key_answer)
        
        # Direct match
        if llm_norm == key_norm:
            return True
        
        # Check if answer key is in LLM's answer
        if key_norm in llm_norm:
            return True
        
        return False
    
    def _compare_multiple_correct_answers(self, llm_answer: str, key_answers: List[str]) -> bool:
        """Compare multiple correct answer responses"""
        import re
        
        # Extract individual choices from LLM answer
        llm_choices = self._extract_choices_from_response(llm_answer)
        
        # Normalize key answers
        key_choices = set(self._normalize_answer(choice) for choice in key_answers)
        
        # Check if extracted choices match the key exactly
        return llm_choices == key_choices
    
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
                        "match": "Error"
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
        matches = sum(1 for r in successful_results if r.get("match") == "Yes")
        total_cost = sum(
            r.get("processing_result", {}).get("metrics", {}).get("total_cost", 0)
            for r in successful_results
        )
        total_tokens = sum(
            r.get("processing_result", {}).get("metrics", {}).get("total_tokens", 0)
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
                "matches": matches,
                "accuracy": matches / len(successful_results) if successful_results else 0,
                "total_cost": total_cost,
                "total_tokens": total_tokens
            },
            "questions": results
        }
        
        # Save to file
        output_file = output_dir / f"jee_questions_{provider}_{model.replace('/', '_')}_response.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Accuracy: {matches}/{len(successful_results)} ({output_data['summary']['accuracy']:.2%})")
        logger.info(f"Total cost: ${total_cost:.4f}")
        
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