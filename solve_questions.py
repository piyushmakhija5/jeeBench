import os
import base64
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
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

def process_single_image(image_path):
    """Process a single image and return the response data"""
    try:
        # Check for API key
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return {
                "error": "ANTHROPIC_API_KEY environment variable is not set",
                "metrics": None,
                "response": None
            }

        # Initialize Anthropic client
        anthropic = Anthropic(api_key=api_key)
        
        # Check if file exists and is readable
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found at path: {image_path}",
                "metrics": None,
                "response": None
            }
        
        if not os.access(image_path, os.R_OK):
            return {
                "error": f"Permission denied when accessing image file: {image_path}",
                "metrics": None,
                "response": None
            }

        # Encode image to base64
        base64_image, encoding_time, encoded_size = encode_image(image_path)

        # Create message with image
        api_call_start = time.time()
        try:
            response = anthropic.messages.create(
                model="claude-3-7-sonnet-latest",
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
        except Exception as api_error:
            return {
                "error": f"API call failed: {str(api_error)}",
                "metrics": None,
                "response": None
            }

        api_call_time = time.time() - api_call_start

        # Return metrics and response
        return {
            "metrics": {
                "encoding_time": encoding_time,
                "encoded_size": encoded_size,
                "api_call_time": api_call_time,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "input_cost": (response.usage.input_tokens / 1000000) * 15,
                "output_cost": (response.usage.output_tokens / 1000000) * 75,
                "total_cost": ((response.usage.input_tokens / 1000000) * 15) + ((response.usage.output_tokens / 1000000) * 75)
            },
            "response": response.content[0].text
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

def process_all_questions():
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

    # Initialize results dictionary
    results = {
        "total_questions": len(questions),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cost": 0,
        "total_tokens": 0,
        "questions": [],
        "errors": []  # New field to track errors
    }

    # Process each question
    for question in tqdm(questions, desc="Processing questions"):
        try:
            # Validate question data
            if not isinstance(question, dict) or "image_path" not in question:
                raise ValueError(f"Invalid question format: {question}")

            # Process the image
            result = process_single_image(question["image_path"])
            
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
            with open("jee_questions_LLM_response.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to file: {str(e)}")

    # Print final summary
    print("\nProcessing complete!")
    print(f"Total questions processed: {len(questions)}")
    print(f"Total tokens used: {results['total_tokens']}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    
    # Print error summary if any errors occurred
    if results["errors"]:
        print("\nErrors encountered:")
        for error in results["errors"]:
            print(f"- Question {error['question']}: {error['error']}")

if __name__ == "__main__":
    process_all_questions()
