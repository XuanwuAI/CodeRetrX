#!/usr/bin/env python3
"""
Script to convert chunk_id to start_line and end_line information
and create a new code_report_convert.json file.
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple

def load_repo_data(repo_file_path: str) -> Dict[str, Any]:
    """Load the repository data file to get chunk line information."""
    print(f"Loading repository data from: {repo_file_path}")
    with open(repo_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_chunk_mapping(repo_data: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """
    Build a mapping from chunk_id (uuid) to (start_line, end_line).
    
    Returns:
        Dict mapping chunk_id to (start_line, end_line) tuple
    """
    chunk_mapping = {}
    
    source_files = repo_data.get('source_files', {})
    for file_path, file_data in source_files.items():
        chunks = file_data.get('chunks', [])
        for chunk in chunks:
            uuid = chunk.get('uuid')
            start_line = chunk.get('start_line')
            end_line = chunk.get('end_line')
            
            if uuid and start_line is not None and end_line is not None:
                chunk_mapping[uuid] = (start_line, end_line)
    
    print(f"Built mapping for {len(chunk_mapping)} chunks")
    return chunk_mapping

def convert_code_report(code_report_path: str, chunk_mapping: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
    """
    Convert code report by adding start_line and end_line information to each location.
    
    Args:
        code_report_path: Path to the original code report JSON file
        chunk_mapping: Mapping from chunk_id to (start_line, end_line)
    
    Returns:
        Converted code report with line information
    """
    print(f"Loading code report from: {code_report_path}")
    with open(code_report_path, 'r', encoding='utf-8') as f:
        code_report = json.load(f)
    
    # Add line information to each location in the code report
    converted_codes = []
    total_locations = 0
    locations_with_line_info = 0
    missing_chunks = []
    
    for code in code_report.get('codes', []):
        converted_code = code.copy()
        converted_locations = []
        
        for location in code.get('locations', []):
            total_locations += 1
            chunk_id = location.get('chunk_id')
            
            if chunk_id in chunk_mapping:
                start_line, end_line = chunk_mapping[chunk_id]
                # Create a copy of the location with added line information
                # Add 1 to convert from 0-based to 1-based indexing
                converted_location = location.copy()
                converted_location['start_line'] = start_line + 1
                converted_location['end_line'] = end_line + 1
                converted_locations.append(converted_location)
                locations_with_line_info += 1
            else:
                print(f"Warning: No line information found for chunk_id: {chunk_id}")
                missing_chunks.append(chunk_id)
                # Keep the original location without line information
                converted_locations.append(location)
        
        converted_code['locations'] = converted_locations
        converted_codes.append(converted_code)
    
    # Create the converted code report
    converted_report = code_report.copy()
    converted_report['codes'] = converted_codes
    
    # Add metadata about the conversion
    converted_report['conversion_info'] = {
        'total_locations': total_locations,
        'locations_with_line_info': locations_with_line_info,
        'missing_chunks': len(missing_chunks),
        'missing_chunk_ids': missing_chunks
    }
    
    print(f"Converted {total_locations} locations")
    print(f"Locations with line info: {locations_with_line_info}")
    print(f"Missing chunks: {len(missing_chunks)}")
    
    return converted_report

def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/convert_chunk_id_to_lines.py <code_report_path> <repo_data_path> <output_path>")
        print("Example: python scripts/convert_chunk_id_to_lines.py code_reports/ollama_ollama/code_report_10_precise_pri.json .data/repos/ollama_ollama.json code_reports/ollama_ollama/code_report_convert.json")
        sys.exit(1)
    
    code_report_path = sys.argv[1]
    repo_data_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Check if input files exist
    if not os.path.exists(code_report_path):
        print(f"Error: Code report file not found: {code_report_path}")
        sys.exit(1)
    
    if not os.path.exists(repo_data_path):
        print(f"Error: Repository data file not found: {repo_data_path}")
        sys.exit(1)
    
    try:
        # Load repository data and build chunk mapping
        repo_data = load_repo_data(repo_data_path)
        chunk_mapping = build_chunk_mapping(repo_data)
        
        # Convert the code report
        converted_report = convert_code_report(code_report_path, chunk_mapping)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the converted report
        print(f"Saving converted report to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
