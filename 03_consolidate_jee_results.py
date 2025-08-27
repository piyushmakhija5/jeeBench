#!/usr/bin/env python3
"""
JEE Benchmark Results Consolidation Script
Consolidates all model results into a comprehensive Excel file for comparison.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def extract_model_name(filename):
    """Extract clean model name from filename."""
    # Remove 'jee_questions_' prefix and '_response.json' suffix
    name = filename.replace('jee_questions_', '').replace('_response.json', '')

    # Clean up model names for better readability
    name_mappings = {
        'openai_gpt-4o': 'GPT-4o',
        'openai_gpt-4.1': 'GPT-4.1',
        'openai_gpt-5': 'GPT-5',
        'openai_o3': 'O3',
        'openai_o4-mini': 'O4-mini',
        'anthropic_claude-opus-4-20250514': 'Claude Opus 4',
        'anthropic_claude-sonnet-4-20250514': 'Claude Sonnet 4',
        'google_gemini-2.0-flash': 'Gemini 2.0 Flash',
        'google_gemini-2.5-pro-preview-06-05': 'Gemini 2.5 Pro',
        'google_gemini-2.5-flash-preview-05-20': 'Gemini 2.5 Flash',
        'xai_grok-4': 'Grok-4',
        'groq_llama-4-maverick': 'LLAMA-4-MAVERICK',
        'groq_llama-4-scout': 'LLAMA-4-SCOUT',
    }

    return name_mappings.get(name, name)

def load_all_results(results_dir):
    """Load all JSON result files from the directory."""
    results_dir = Path(results_dir)
    all_results = {}

    for json_file in results_dir.glob('*.json'):
        if json_file.name == '.DS_Store':
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = extract_model_name(json_file.name)
                all_results[model_name] = data
                print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    return all_results

def create_summary_comparison(all_results):
    """Create a summary comparison of all models."""
    summary_data = []

    # Define the topper score thresholds based on the CRL data
    def get_air_ranking(total_score):
        """Determine All India Rank based on total score compared to human toppers."""
        if total_score > 332:
            return "> AIR 1"  # Better than rank 1
        elif total_score == 332:
            return "AIR 1-2"  # Tie with top 2
        elif total_score > 330:
            return "> AIR 3"  # Better than rank 3
        elif total_score == 330:
            return "AIR 3"
        elif total_score > 327:
            return "> AIR 4"  # Better than rank 4
        elif total_score == 327:
            return "AIR 4"
        elif total_score > 324:
            return "> AIR 5"  # Better than rank 5
        elif total_score == 324:
            return "AIR 5"
        elif total_score > 321:
            return "> AIR 6"  # Better than rank 6-7
        elif total_score == 321:
            return "AIR 6-7"
        elif total_score > 319:
            return "> AIR 8"  # Better than rank 8-9
        elif total_score == 319:
            return "AIR 8-9"
        elif total_score > 317:
            return "> AIR 10"  # Better than rank 10
        elif total_score == 317:
            return "AIR 10"
        else:
            return "< AIR 10"  # Below top 10

    for model_name, data in all_results.items():
        summary = data.get('summary', {})
        total_score = summary.get('total_score', 0)

        # Calculate total API call time, average per question, and collect all API times for percentiles
        total_api_time = 0
        total_questions = data.get('total_questions', 0)
        api_times = []

        for question in data.get('questions', []):
            processing_result = question.get('processing_result', {})
            if processing_result is not None:
                metrics = processing_result.get('metrics', {})
                if metrics is not None:
                    api_call_time = metrics.get('api_call_time', 0)
                    total_api_time += api_call_time
                    api_times.append(api_call_time)

        avg_api_time_per_question = total_api_time / total_questions if total_questions > 0 else 0
        total_api_time_minutes = total_api_time / 60  # Convert seconds to minutes

        # Calculate median, P75, and P90 response times
        if api_times:
            median_response_time = np.median(api_times)
            p75_response_time = np.percentile(api_times, 75)
            p90_response_time = np.percentile(api_times, 90)
        else:
            median_response_time = p75_response_time = p90_response_time = 0

        # Calculate average tokens per question
        total_tokens = summary.get('total_tokens', 0)
        avg_tokens_per_question = total_tokens / total_questions if total_questions > 0 else 0

        row = {
            'Model': model_name,
            'Total Score': total_score,
            'All India Rank (AIR)': get_air_ranking(total_score),
            'Score Percentage (%)': round(summary.get('score_percentage', 0), 2),
            'Total Cost ($)': round(summary.get('total_cost', 0), 4),
            'Avg API Time per Question (s)': round(avg_api_time_per_question, 2),
            'Median Response Time (s)': round(median_response_time, 2),
            'P75 Response Time (s)': round(p75_response_time, 2),
            'P90 Response Time (s)': round(p90_response_time, 2),
            'Avg Tokens per Question': round(avg_tokens_per_question, 0),
            'Total API Call Time (min)': round(total_api_time_minutes, 2),
            'Total Questions': data.get('total_questions', 0),
            'Successfully Processed': summary.get('successful_processing', 0),
            'Correct Answers (Yes)': summary.get('yes_matches', 0),
            'Partial Answers': summary.get('partial_matches', 0),
            'Wrong Answers (No)': summary.get('no_matches', 0),
            'Input Tokens': summary.get('input_tokens', 0),
            'Output Tokens': summary.get('output_tokens', 0),
            'Total Tokens': summary.get('total_tokens', 0),
            'Provider': data.get('provider', ''),

        }
        summary_data.append(row)

    # Sort by score percentage descending
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Score Percentage (%)', ascending=False)

    return summary_df

def create_unit_wise_analysis(all_results):
    """Create unit-wise performance analysis for all subjects with models as columns."""
    # First, determine model order based on overall score percentage (descending)
    model_performance = []
    for model_name, data in all_results.items():
        summary = data.get('summary', {})
        score_percentage = summary.get('score_percentage', 0)
        model_performance.append((model_name, score_percentage))

    # Sort models by score percentage descending
    model_performance.sort(key=lambda x: x[1], reverse=True)
    ordered_models = [model[0] for model in model_performance]

    # Get all unique subject-unit combinations and count questions
    all_units = {}
    for model_name, data in all_results.items():
        for question in data.get('questions', []):
            subject = question['metadata']['subject']
            unit = question['metadata']['unit']
            unit_key = f"{subject} - {unit}"

            if unit_key not in all_units:
                all_units[unit_key] = {
                    'subject': subject,
                    'unit': unit,
                    'total_questions': 0
                }
            all_units[unit_key]['total_questions'] += 1
        break  # Only need to count once since all models have same questions

    # For each unit, collect stats for all models
    unit_rows = []

    for unit_key, unit_info in all_units.items():
        row = {
            'Subject': unit_info['subject'],
            'Unit': unit_info['unit'],
            'Number of Questions': unit_info['total_questions']
        }

        # For each model (in performance order), calculate correct percentage for this unit
        for model_name in ordered_models:
            data = all_results[model_name]
            total_questions = 0
            correct_answers = 0

            # Calculate stats from questions for this specific unit
            for question in data.get('questions', []):
                if (question['metadata']['subject'] == unit_info['subject'] and
                    question['metadata']['unit'] == unit_info['unit']):
                    total_questions += 1
                    if question.get('match') == 'Yes':
                        correct_answers += 1

            # Calculate correct percentage
            correct_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0

            # Add column for this model (just the percentage)
            row[f"{model_name} (%)"] = round(correct_percentage, 2)

        unit_rows.append(row)

    # Sort by subject, then unit
    unit_df = pd.DataFrame(unit_rows)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(['Subject', 'Unit'], ascending=[True, True])

    return unit_df

def create_subject_analysis(all_results):
    """Create subject-wise performance analysis with subjects in rows and models in columns."""
    # First, determine model order based on overall score percentage (descending)
    model_performance = []
    for model_name, data in all_results.items():
        summary = data.get('summary', {})
        score_percentage = summary.get('score_percentage', 0)
        model_performance.append((model_name, score_percentage))

    # Sort models by score percentage descending
    model_performance.sort(key=lambda x: x[1], reverse=True)
    ordered_models = [model[0] for model in model_performance]

    # Get all unique subjects and calculate total questions and max score
    all_subjects = {}
    for model_name, data in all_results.items():
        for question in data.get('questions', []):
            subject = question['metadata']['subject']
            if subject not in all_subjects:
                all_subjects[subject] = {
                    'total_questions': 0,
                    'max_score': 0
                }
            all_subjects[subject]['total_questions'] += 1
            all_subjects[subject]['max_score'] += question['metadata']['scoring']['full_marks']
        break  # Only need to count once since all models have same questions

    # For each subject, collect stats for all models
    subject_rows = []

    for subject, subject_info in all_subjects.items():
        row = {
            'Subject': subject,
            'Total Questions': subject_info['total_questions'],
            'Max Score': subject_info['max_score']
        }

        # For each model (in performance order), calculate score percentage for this subject
        for model_name in ordered_models:
            data = all_results[model_name]
            total_score = 0
            max_possible_score = 0

            # Calculate stats from questions for this specific subject
            for question in data.get('questions', []):
                if question['metadata']['subject'] == subject:
                    total_score += question.get('score', 0)
                    max_possible_score += question['metadata']['scoring']['full_marks']

            # Calculate score percentage
            score_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0

            # Add column for this model (score percentage)
            row[f"{model_name} (%)"] = round(score_percentage, 2)

        subject_rows.append(row)

    # Sort by subject name
    subject_df = pd.DataFrame(subject_rows)
    if not subject_df.empty:
        subject_df = subject_df.sort_values(['Subject'], ascending=[True])

    return subject_df

def create_difficulty_analysis(all_results):
    """Create difficulty-wise performance analysis with subject-difficulty in rows and models in columns."""
    # First, determine model order based on overall score percentage (descending)
    model_performance = []
    for model_name, data in all_results.items():
        summary = data.get('summary', {})
        score_percentage = summary.get('score_percentage', 0)
        model_performance.append((model_name, score_percentage))

    # Sort models by score percentage descending
    model_performance.sort(key=lambda x: x[1], reverse=True)
    ordered_models = [model[0] for model in model_performance]

    # Get all unique subject-difficulty combinations and calculate total questions and max score
    all_difficulties = {}
    for model_name, data in all_results.items():
        for question in data.get('questions', []):
            subject = question['metadata']['subject']
            difficulty = question['metadata']['difficulty']
            difficulty_key = f"{subject} - {difficulty}"

            if difficulty_key not in all_difficulties:
                all_difficulties[difficulty_key] = {
                    'subject': subject,
                    'difficulty': difficulty,
                    'total_questions': 0,
                    'max_score': 0
                }
            all_difficulties[difficulty_key]['total_questions'] += 1
            all_difficulties[difficulty_key]['max_score'] += question['metadata']['scoring']['full_marks']
        break  # Only need to count once since all models have same questions

    # For each difficulty, collect stats for all models
    difficulty_rows = []

    for difficulty_key, difficulty_info in all_difficulties.items():
        row = {
            'Subject': difficulty_info['subject'],
            'Difficulty': difficulty_info['difficulty'],
            'Total Questions': difficulty_info['total_questions'],
            'Max Score': difficulty_info['max_score']
        }

        # For each model (in performance order), calculate score percentage for this subject-difficulty
        for model_name in ordered_models:
            data = all_results[model_name]
            total_score = 0
            max_possible_score = 0

            # Calculate stats from questions for this specific subject-difficulty
            for question in data.get('questions', []):
                if (question['metadata']['subject'] == difficulty_info['subject'] and
                    question['metadata']['difficulty'] == difficulty_info['difficulty']):
                    total_score += question.get('score', 0)
                    max_possible_score += question['metadata']['scoring']['full_marks']

            # Calculate score percentage
            score_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0

            # Add column for this model (score percentage)
            row[f"{model_name} (%)"] = round(score_percentage, 2)

        difficulty_rows.append(row)

    # Sort by subject, then difficulty
    difficulty_df = pd.DataFrame(difficulty_rows)
    if not difficulty_df.empty:
        difficulty_df = difficulty_df.sort_values(['Subject', 'Difficulty'], ascending=[True, True])

    return difficulty_df

def create_cost_efficiency_analysis(all_results):
    """Create cost efficiency analysis."""
    cost_data = []

    for model_name, data in all_results.items():
        summary = data.get('summary', {})

        total_cost = summary.get('total_cost', 0)
        accuracy = summary.get('accuracy', 0)
        total_score = summary.get('total_score', 0)
        total_tokens = summary.get('total_tokens', 0)
        total_questions = data.get('total_questions', 1)  # Use total_questions as denominator

        # Calculate efficiency metrics
        cost_per_question = total_cost / total_questions
        cost_per_correct = total_cost / summary.get('yes_matches', 1) if summary.get('yes_matches', 0) > 0 else float('inf')
        cost_per_point = total_cost / total_score if total_score > 0 else float('inf')
        tokens_per_question = total_tokens / total_questions

        cost_data.append({
            'Model': model_name,
            'Total Cost ($)': round(total_cost, 4),
            'Cost per Question ($)': round(cost_per_question, 4),
            'Cost per Correct Answer ($)': round(cost_per_correct, 4) if cost_per_correct != float('inf') else 'N/A',
            'Cost per Point ($)': round(cost_per_point, 4) if cost_per_point != float('inf') else 'N/A',
            'Tokens per Question': round(tokens_per_question, 0),
            'Accuracy (%)': round(accuracy * 100, 2),
            'Score Percentage (%)': round(summary.get('score_percentage', 0), 2)
        })

    return pd.DataFrame(cost_data)

def main():
    """Main function to consolidate all results."""
    # Path to the results directory
    results_dir = "./data/outputs/results"

    print("Loading JEE benchmark results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        print("No results found!")
        return

    print(f"Loaded {len(all_results)} model results")

    # Create different analysis sheets
    print("Creating summary comparison...")
    summary_df = create_summary_comparison(all_results)

    print("Creating subject analysis...")
    subject_df = create_subject_analysis(all_results)

    print("Creating unit-wise analysis...")
    unit_df = create_unit_wise_analysis(all_results)

    print("Creating difficulty analysis...")
    difficulty_df = create_difficulty_analysis(all_results)

    print("Creating cost efficiency analysis...")
    cost_df = create_cost_efficiency_analysis(all_results)

    # Create Excel file with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = f"./data/outputs/results/jee_benchmark_consolidated_{timestamp}.xlsx"

    print(f"Writing to Excel file: {output_file}")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write each sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        subject_df.to_excel(writer, sheet_name='Subject_Analysis', index=False)
        unit_df.to_excel(writer, sheet_name='Unit_Analysis', index=False)
        difficulty_df.to_excel(writer, sheet_name='Difficulty_Analysis', index=False)
        cost_df.to_excel(writer, sheet_name='Cost_Efficiency', index=False)

        # Get workbook and add formatting
        workbook = writer.book

        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # Format each sheet
        df_mapping = {
            'Summary': summary_df,
            'Unit_Analysis': unit_df,
            'Subject_Analysis': subject_df,
            'Difficulty_Analysis': difficulty_df,
            'Cost_Efficiency': cost_df
        }

        for sheet_name in ['Summary', 'Unit_Analysis', 'Subject_Analysis', 'Difficulty_Analysis', 'Cost_Efficiency']:
            worksheet = writer.sheets[sheet_name]
            current_df = df_mapping[sheet_name]

            # Format headers
            for col_num, value in enumerate(current_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Auto-fit columns
            for i, col in enumerate(current_df.columns):
                column_len = max(
                    current_df[col].astype(str).str.len().max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(i, i, min(column_len, 50))

    print(f"✅ Successfully created consolidated Excel file: {output_file}")
    print("\nSummary of results:")
    print(summary_df[['Model', "Total Score", "All India Rank (AIR)", "Score Percentage (%)", 'Total Cost ($)', "Avg API Time per Question (s)", "Median Response Time (s)", "P75 Response Time (s)", "P90 Response Time (s)", "Avg Tokens per Question"]].to_string(index=False))

if __name__ == "__main__":
    main()