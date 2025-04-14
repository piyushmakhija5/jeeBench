import os
import base64
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm

# Load environment variables
load_dotenv()

def encode_image(image_path):
    """Convert image to base64 string"""
    try:
        start_time = time.time()
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        encoding_time = time.time() - start_time
        encoded_size = len(encoded)
        return encoded, encoding_time, encoded_size
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing image file: {image_path}")
    except Exception as e:
        raise Exception(f"Error encoding image {image_path}: {str(e)}")

def get_model_config(provider):
    """Get model configuration based on provider"""
    configs = {
        "anthropic": {
            "model": "claude-3-7-sonnet-latest",
            "input_price": 15/1000000,  # $15 per 1M tokens
            "output_price": 75/1000000,  # $75 per 1M tokens
        },
        "openai": {
            "model": "gpt-4.5-preview",
            "input_price": 75/1000000,    # 2.5 per 1M tokens
            "output_price": 150/1000000,   # $10 per 1M tokens
        },
        "google": {
            "model": "gemini-2.5-pro-exp-03-25",
            "input_price":  0,    # $1.25 per 1K tokens
            "output_price": 0,   # $1.25 per 1K tokens
        }
    }
    return configs.get(provider.lower())

def process_single_image(image_path, provider="anthropic"):
    """Process a single image and return the response data"""
    try:
        config = get_model_config(provider)
        if not config:
            return {
                "error": f"Invalid provider: {provider}. Supported providers are: anthropic, openai, google",
                "metrics": None,
                "response": None
            }

        # Check for appropriate API key
        api_key = os.getenv(f'{provider.upper()}_API_KEY')
        if not api_key:
            return {
                "error": f"{provider.upper()}_API_KEY environment variable is not set",
                "metrics": None,
                "response": None
            }

        # Encode image to base64
        base64_image, encoding_time, encoded_size = encode_image(image_path)

        api_call_start = time.time()
        response = None
        usage = None

        if provider.lower() == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=config["model"],
                max_tokens=5000,
                temperature=0.0,
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
                        {
                            "type": "text",
                            "text": "Analyze this question and provide your response in the following JSON format:\n{\n  \"Solution\": \"Detailed step-by-step solution explanation\",\n  \"Final Answer\": \"MCQ answer choice(s) or the final numerical answer\"\n}\nONLY provide valid JSON structure and escape any special characters properly."
                        }
                    ]
                }]
            )
            usage = response.usage
            response_text = response.content[0].text

        elif provider.lower() == "openai":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Analyze this question and provide your response in the following JSON format:\n{\n  \"Solution\": \"Detailed step-by-step solution explanation\",\n  \"Final Answer\": \"MCQ answer choice(s) or the final numerical answer\"\n}\nONLY provide valid JSON structure and escape any special characters properly."
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.0
            )
            usage = response.usage
            response_text = response.choices[0].message.content

        elif provider.lower() == "google":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config["model"])
            response = model.generate_content(
                contents=[
                    {
                        "parts": [
                            {"mime_type": "image/png", "data": base64.b64decode(base64_image)},
                            {"text": "Analyze this question and provide your response in the following JSON format:\n{\n  \"Solution\": \"Detailed step-by-step solution explanation\",\n  \"Final Answer\": \"MCQ answer choice(s) or the final numerical answer\"\n}\nONLY provide valid JSON structure and escape any special characters properly."}
                        ]
                    }
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    candidate_count=1,
                )
            )
            # Note: Gemini doesn't provide detailed token usage
            usage = {"prompt_tokens": 0, "completion_tokens": 0}  # Placeholder
            response_text = response.text

        api_call_time = time.time() - api_call_start

        # Calculate costs based on provider's pricing
        input_cost = (usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0) * config["input_price"]
        output_cost = (usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0) * config["output_price"]
        total_tokens = (usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0) + (usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0)

        return {
            "metrics": {
                "encoding_time": encoding_time,
                "encoded_size": encoded_size,
                "api_call_time": api_call_time,
                "input_tokens": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                "output_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                "total_tokens": total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            },
            "response": response_text
        }

    except Exception as e:
        return {
            "error": f"Unexpected error processing image {image_path}: {str(e)}",
            "metrics": None,
            "response": None
        }

def normalize_answer(answer):
    """Normalize answer string for comparison by removing common formatting"""
    if answer is None:
        return None
    # Convert to string and uppercase
    answer = str(answer).upper()
    # Remove common decorators like parentheses, periods, spaces
    answer = answer.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')
    return answer

def answers_match(llm_answer, key_answer):
    """
    Intelligently compare LLM answer with answer key
    Returns True if answers match, False otherwise
    """
    if llm_answer is None or key_answer is None:
        return False

    # Normalize both answers
    llm_norm = normalize_answer(llm_answer)
    key_norm = normalize_answer(key_answer)

    # Direct match after normalization
    if llm_norm == key_norm:
        return True

    # Check if answer key is in LLM's answer
    # This handles cases like "(B) -17/4" matching with "B"
    if key_norm in llm_norm:
        return True

    return False

def process_all_questions(provider="anthropic", force_override=False):
    # Load question metadata
    try:
        if not os.path.exists("question_metadata.json"):
            print("Error: question_metadata.json file not found!")
            return

        with open("question_metadata.json", "r") as f:
            questions = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in question_metadata.json: {str(e)}")
        return
    except Exception as e:
        print(f"Error loading question metadata: {str(e)}")
        return

    if not questions:
        print("Error: No questions found in metadata file")
        return

    # Get model configuration
    config = get_model_config(provider)
    if not config:
        print(f"Error: Invalid provider: {provider}")
        return

    # Define output filename
    output_filename = f"jee_questions_{provider}_{config['model']}_response.json"

    # Load existing results if they exist and we're not forcing override
    existing_results = {}
    processed_questions = set()
    if not force_override and os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                existing_results = json.load(f)
                # Create a set of already processed question paths
                processed_questions = {q["metadata"]["image_path"] for q in existing_results.get("questions", [])}
                print(f"Found {len(processed_questions)} previously processed questions")
        except Exception as e:
            print(f"Warning: Could not load existing results: {str(e)}")
            existing_results = {}
            processed_questions = set()

    # Initialize results dictionary
    results = existing_results if not force_override and existing_results else {
        "total_questions": len(questions),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cost": 0,
        "total_tokens": 0,
        "provider": provider,
        "model": config["model"],
        "questions": [],
        "errors": []
    }

    # Filter questions that need processing
    questions_to_process = [q for q in questions if force_override or q["image_path"] not in processed_questions]

    if not questions_to_process:
        print("All questions have already been processed. Use --force to reprocess all questions.")
        return

    print(f"Processing {len(questions_to_process)} questions...")

    # Process each question
    for question in tqdm(questions_to_process, desc="Processing questions"):
        try:
            # Validate question data
            if not isinstance(question, dict) or "image_path" not in question:
                raise ValueError(f"Invalid question format: {question}")

            # Process the image
            result = process_single_image(question["image_path"], provider)

            # Check for processing errors
            if "error" in result and result["error"]:
                results["errors"].append({
                    "question": question["image_path"],
                    "error": result["error"]
                })
                continue

            # Extract Final Answer from LLM response
            final_answer = None
            if result["response"]:
                try:
                    llm_response = json.loads(result["response"])
                    final_answer = llm_response.get("Final Answer")
                except json.JSONDecodeError as e:
                    results["errors"].append({
                        "question": question["image_path"],
                        "error": f"Failed to parse LLM response as JSON: {str(e)}"
                    })

            # Compare answers and determine if they match
            answer_match = answers_match(final_answer, question.get("answer_key"))

            question_result = {
                "metadata": question,
                "processing_result": result,
                "final_answer": final_answer,
                "match": "Yes" if answer_match else "No"
            }

            # Update totals if processing was successful
            if result["metrics"]:
                results["total_cost"] += result["metrics"]["total_cost"]
                results["total_tokens"] += result["metrics"]["total_tokens"]

            results["questions"].append(question_result)

        except Exception as e:
            results["errors"].append({
                "question": question.get("image_path", "Unknown"),
                "error": f"Unexpected error processing question: {str(e)}"
            })

        # Save results after each question
        try:
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to file: {str(e)}")

    # Print final summary
    print("\nProcessing complete!")
    print(f"Total questions processed in this run: {len(questions_to_process)}")
    print(f"Total tokens used: {results['total_tokens']}")
    print(f"Total cost: ${results['total_cost']:.4f}")

    # Print error summary if any errors occurred
    if results["errors"]:
        print("\nErrors encountered:")
        for error in results["errors"]:
            print(f"- Question {error['question']}: {error['error']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process questions using different AI model providers')
    parser.add_argument('--provider', type=str, default='anthropic',
                      choices=['anthropic', 'openai', 'google'],
                      help='The AI model provider to use (default: anthropic)')
    parser.add_argument('--force', '-f', action='store_true',
                      help='Force reprocessing of all questions, ignoring existing results')
    args = parser.parse_args()

    process_all_questions(args.provider, force_override=args.force)
