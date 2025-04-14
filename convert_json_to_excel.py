import json
import pandas as pd
import sys
import os

def convert_json_to_excel(input_file, output_file):
    try:
        # Read the JSON data from file
        with open(input_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            
        # Create a list to store the structured data
        excel_data = []
        
        for question in json_data['questions']:
            metadata = question['metadata']
            metrics = question['processing_result']['metrics'] if 'processing_result' in question else {}
            
            # Extract the response and model's final answer
            response = ""
            model_final_answer = ""
            if 'processing_result' in question and 'response' in question['processing_result']:
                raw_response = question['processing_result']['response']
                # Try to extract structured data from the response
                if "```json" in raw_response:
                    try:
                        json_start = raw_response.find('{')
                        json_end = raw_response.rfind('}')
                        if json_start != -1 and json_end != -1:
                            json_content = raw_response[json_start:json_end+1]
                            response_json = json.loads(json_content)
                            if "Final Answer" in response_json:
                                model_final_answer = response_json["Final Answer"]
                    except:
                        model_final_answer = "Error parsing response"
            
            # Create a row for the Excel file
            row = {
                'Subject': metadata.get('subject', ''),
                'Question Number': metadata.get('question_number', ''),
                'Page Number': metadata.get('page_number', ''),
                'Unit': metadata.get('unit', ''),
                'Image Path': metadata.get('image_path', ''),
                'Expected Answer': metadata.get('answer', ''),
                'Answer Value': metadata.get('answer_value', ''),
                'Model Final Answer': model_final_answer,
                'Match': question.get('match', ''),
                'Input Tokens': metrics.get('input_tokens', ''),
                'Output Tokens': metrics.get('output_tokens', ''),
                'Total Tokens': metrics.get('total_tokens', ''),
                'Cost': metrics.get('total_cost', ''),
                'Response Time (s)': metrics.get('api_call_time', '')
            }
            
            excel_data.append(row)
        
        # Create a DataFrame
        df = pd.DataFrame(excel_data)
        
        # Add a summary row
        summary = {
            'Subject': 'SUMMARY',
            'Question Number': '',
            'Page Number': '',
            'Unit': '',
            'Image Path': '',
            'Expected Answer': '',
            'Answer Value': '',
            'Model Final Answer': '',
            'Match': f"Matches: {df['Match'].value_counts().get('Yes', 0)}/{len(df)}",
            'Input Tokens': df['Input Tokens'].sum(),
            'Output Tokens': df['Output Tokens'].sum(),
            'Total Tokens': df['Total Tokens'].sum(),
            'Cost': df['Cost'].sum(),
            'Response Time (s)': df['Response Time (s)'].sum()
        }
        
        # Add summary at the end
        df_summary = pd.DataFrame([summary])
        df_final = pd.concat([df, df_summary], ignore_index=True)
        
        # Get input file base name without extension
        input_base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create the Excel file with only the results sheet
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name=f'{input_base_name}_Results', index=False)
            
        print(f"Excel file '{output_file}' has been created with the Results sheet.")
        print(f"Total Questions: {json_data['total_questions']}")
        print(f"Questions Processed: {len(df)}")
        print(f"Total Cost: ${json_data['total_cost']:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error converting JSON to Excel: {e}")
        return False

# If running as a script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python json_to_excel.py <input_json_file> <output_excel_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = convert_json_to_excel(input_file, output_file)
    if not success:
        sys.exit(1)