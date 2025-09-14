#!/usr/bin/env python3
"""
Script to merge code_report data into dataset_vuln.json
This script takes the codes from code_report and adds them as snippets to the corresponding queries in dataset_vuln.json
"""

import json
import sys
import os
from typing import Dict, Any, List

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content."""
    print(f"Loading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_matching_query(dataset_queries: List[Dict], code_type: str) -> Dict[str, Any]:
    """
    Find the query in dataset that matches the code_type from code_report.
    This is done by comparing the filter_prompt with the code_type.
    """
    for query in dataset_queries:
        if query.get('filter_prompt') == code_type:
            return query
    return None

def merge_code_report_to_dataset(code_report_path: str, dataset_path: str, output_path: str = None):
    """
    Merge code_report codes into dataset_vuln.json
    
    Args:
        code_report_path: Path to the code_report JSON file
        dataset_path: Path to the dataset_vuln.json file
        output_path: Path for the output file (defaults to dataset_path with _merged suffix)
    """
    if output_path is None:
        base_name = os.path.splitext(dataset_path)[0]
        output_path = f"{base_name}_merged.json"
    
    # Load both files
    code_report = load_json_file(code_report_path)
    dataset = load_json_file(dataset_path)
    
    print(f"Found {len(code_report.get('codes', []))} codes in code_report")
    print(f"Found {len(dataset.get('queries', []))} queries in dataset")
    
    # Extract cost information from code_report
    cost_info = code_report.get('cost_info', {})
    prompt_costs = cost_info.get('prompt_costs', [])
    
    # Create a mapping from code_type to cost
    code_type_to_cost = {}
    for prompt_cost in prompt_costs:
        prompt = prompt_cost.get('prompt', '')
        cost = prompt_cost.get('cost', 0.0)
        # The prompt in cost_info is truncated, so we need to match by prefix
        code_type_to_cost[prompt] = cost
    
    # Also create a mapping by matching the beginning of code_type with prompt
    for code in code_report.get('codes', []):
        code_type = code.get('code_type', '')
        if code_type:
            # Find matching cost by comparing the beginning of code_type with prompt
            for prompt_cost in prompt_costs:
                prompt = prompt_cost.get('prompt', '')
                cost = prompt_cost.get('cost', 0.0)
                # Check if code_type starts with the prompt (removing the "..." suffix)
                if prompt.endswith('...') and code_type.startswith(prompt[:-3]):
                    code_type_to_cost[code_type] = cost
                    break
                elif code_type == prompt:
                    code_type_to_cost[code_type] = cost
                    break
    
    # Process each code in the code_report
    merged_count = 0
    skipped_count = 0
    
    for code in code_report.get('codes', []):
        code_type = code.get('code_type', '')
        locations = code.get('locations', [])
        
        if not code_type or not locations:
            print(f"Skipping code with empty code_type or locations")
            skipped_count += 1
            continue
        
        # Find matching query in dataset
        matching_query = find_matching_query(dataset.get('queries', []), code_type)
        
        if matching_query:
            # Get cost for this code type
            cost = code_type_to_cost.get(code_type, 0.0)
            
            # Convert locations to snippets format
            snippets = []
            for location in locations:
                snippet = {
                    "name": location.get('name', ''),
                    "type": location.get('type', ''),
                    "file_path": location.get('file_path', ''),
                    "start_line": location.get('start_line'),
                    "end_line": location.get('end_line')
                }
                # Only add if we have the required fields
                if snippet['name'] and snippet['file_path'] and snippet['start_line'] and snippet['end_line']:
                    snippets.append(snippet)
            
            if snippets:
                # Add snippets to the matching query
                if 'snippets' not in matching_query:
                    matching_query['snippets'] = []
                
                # Add new snippets (avoid duplicates)
                existing_snippets = matching_query['snippets']
                for snippet in snippets:
                    # Check if snippet already exists (by name, file_path, and start_line)
                    if not any(
                        existing['name'] == snippet['name'] and 
                        existing['file_path'] == snippet['file_path'] and 
                        existing['start_line'] == snippet['start_line']
                        for existing in existing_snippets
                    ):
                        existing_snippets.append(snippet)
                
                # Add or update cost information
                matching_query['cost'] = cost
                
                print(f"Added {len(snippets)} snippets to query: {matching_query.get('name', 'Unknown')} (cost: {cost})")
                merged_count += 1
            else:
                print(f"No valid snippets found for code_type: {code_type[:100]}...")
                skipped_count += 1
        else:
            print(f"No matching query found for code_type: {code_type[:100]}...")
            skipped_count += 1
    
    # Save the merged dataset
    print(f"Saving merged dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerge completed!")
    print(f"Successfully merged: {merged_count} codes")
    print(f"Skipped: {skipped_count} codes")
    
    # Print summary of snippets per query
    print(f"\nSnippets per query:")
    for query in dataset.get('queries', []):
        snippet_count = len(query.get('snippets', []))
        if snippet_count > 0:
            print(f"  - {query.get('name', 'Unknown')}: {snippet_count} snippets")

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/merge_code_report_to_dataset.py <code_report_path> <dataset_path> [output_path]")
        print("Example: python scripts/merge_code_report_to_dataset.py code_reports/ollama_ollama/code_report_10_precise_pri_convert.json bench/dataset_vuln.json")
        sys.exit(1)
    
    code_report_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Check if input files exist
    if not os.path.exists(code_report_path):
        print(f"Error: Code report file not found: {code_report_path}")
        sys.exit(1)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    try:
        merge_code_report_to_dataset(code_report_path, dataset_path, output_path)
    except Exception as e:
        print(f"Error during merge: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
