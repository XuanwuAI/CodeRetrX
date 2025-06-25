import json
import sys
from typing import Dict, Set, List, Tuple, Optional

def load_json_data(filename: str) -> Dict:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filename}: {e}")
        sys.exit(1)

def extract_bug_ids(data: Dict) -> Set[str]:
    ids = set()
    for bug in data.get('bugs', []):
        for location in bug.get('locations', []):
            ids.add(location.get('id', ''))
    return ids

def extract_timing_info(data: Dict) -> Optional[Dict]:
    """Extract timing information from bug report data"""
    return data.get('timing_info')

def analyze_datasets(adaptive_data: Dict, precise_data: Dict) -> Tuple[int, int, int, int, Optional[Dict], Optional[Dict]]:
    """
    Analyze the relationship between different methods.
    
    Returns:
        - fast_method_count: Total number of items in fast/balance/adaptive dataset
        - ground_truth_count: Total number of items in precise dataset
        - contained_count: Number of fast/balance/adaptive items contained in precise dataset
        - not_contained_count: Number of fast/balance/adaptive items not contained in precise dataset
        - other_timing: Timing info from other data
        - precise_timing: Timing info from precise data
    """
    method_ids = extract_bug_ids(adaptive_data)
    precise_ids = extract_bug_ids(precise_data)
    
    # Remove empty IDs
    method_ids.discard('')
    precise_ids.discard('')
    
    # Calculate intersections
    contained_ids = method_ids.intersection(precise_ids)
    not_contained_ids = method_ids - precise_ids
    
    # Extract timing information
    other_timing = extract_timing_info(other_data)
    precise_timing = extract_timing_info(precise_data)
    
    return len(method_ids), len(precise_ids), len(contained_ids), len(not_contained_ids), other_timing, precise_timing

def print_timing_comparison(other_timing: Optional[Dict], precise_timing: Optional[Dict], method_name: str):
    """Print timing comparison between adaptive/balance and precise methods"""
    print(f"\n--- Timing Comparison for {method_name} ---")
    
    if other_timing and precise_timing:
        other_total = other_timing.get('total_duration', 0)
        precise_total = precise_timing.get('total_duration', 0)
        
        print(f"Adaptive/Balance total time: {other_total:.2f} seconds")
        print(f"Precise total time: {precise_total:.2f} seconds")
        
        if precise_total > 0:
            speedup = precise_total / other_total if other_total > 0 else float('inf')
            print(f"Speedup: {speedup:.2f}x")
        
        # Show average prompt times
        other_avg = other_timing.get('average_prompt_duration', 0)
        precise_avg = precise_timing.get('average_prompt_duration', 0)
        
        print(f"Average prompt time - Adaptive/Balance: {other_avg:.2f}s, Precise: {precise_avg:.2f}s")
        
    elif other_timing:
        other_total = other_timing.get('total_duration', 0)
        print(f"Adaptive/Balance total time: {other_total:.2f} seconds")
        print("Precise timing data not available")
    elif precise_timing:
        precise_total = precise_timing.get('total_duration', 0)
        print(f"Precise total time: {precise_total:.2f} seconds")
        print("Adaptive/Balance timing data not available")
    else:
        print("No timing data available for comparison")

def print_analysis(filename_prefix: str, method_count: int, precise_count: int, 
                  contained_count: int, not_contained_count: int, 
                  other_timing: Optional[Dict] = None, precise_timing: Optional[Dict] = None):
    print(f"\n=== Analysis for {filename_prefix} ===")
    print(f"Total adaptive/balance items: {method_count}")
    print(f"Total precise items: {precise_count}")
    print(f"Adaptive items contained in precise: {contained_count}")
    print(f"Adaptive items NOT contained in precise: {not_contained_count}")    
    
    # Calculate proportions
    if method_count > 0:
        contained_proportion = contained_count / method_count
        not_contained_proportion = not_contained_count / method_count
        print(f"\nProportions:")
        print(f"Contained proportion: {contained_proportion:.4f} ({contained_proportion*100:.2f}%)")
        print(f"Not contained proportion: {not_contained_proportion:.4f} ({not_contained_proportion*100:.2f}%)")
    
    # Calculate ratio of contained to total precise
    if precise_count > 0:
        contained_to_precise_ratio = contained_count / precise_count
        print(f"\nRatio of contained items to total precise items: {contained_to_precise_ratio:.4f} ({contained_to_precise_ratio*100:.2f}%)")
    
    # Add timing comparison
    print_timing_comparison(other_timing, precise_timing, filename_prefix)
    
    print("-" * 60)

def print_summary(results: List[Tuple[str, int, int, int, int, Optional[Dict], Optional[Dict]]]):
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    headers = ["Dataset", "Other", "Precise", "Contained", "Coverage %", "Time (A)", "Time (P)", "Speedup"]
    print(f"{headers[0]:<30} {headers[1]:<9} {headers[2]:<8} {headers[3]:<10} {headers[4]:<11} {headers[5]:<9} {headers[6]:<9} {headers[7]:<8}")
    print("-" * 100)
    
    for desc, method_count, precise_count, contained_count, not_contained_count, other_timing, precise_timing in results:
        coverage_pct = (contained_count / precise_count * 100) if precise_count > 0 else 0
        
        # Extract timing information
        other_time = other_timing.get('total_duration', 0) if other_timing else 0
        precise_time = precise_timing.get('total_duration', 0) if precise_timing else 0
        
        speedup = precise_time / other_time if other_time > 0 and precise_time > 0 else 0
        
        # Format time strings
        other_time_str = f"{other_time:.0f}s" if other_time > 0 else "N/A"
        precise_time_str = f"{precise_time:.0f}s" if precise_time > 0 else "N/A"
        speedup_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        
        print(f"{desc:<30} {method_count:<9} {precise_count:<8} {contained_count:<10} {coverage_pct:>9.1f}% {adaptive_time_str:>8} {precise_time_str:>8} {speedup_str:>7}")



def main():
    # Define the bug reports folder path
    bug_reports_folder = "bug_reports/"
    
    # Define the datasets to analyze
    datasets = [
        ("bug_report_adaptive_sentence_True_one.json", "bug_report_precise_sentence_True.json", "adaptive_sentence_True_one"),
        ("bug_report_adaptive_sentence_True_sec.json", "bug_report_precise_sentence_True.json", "adaptive_sentence_True_sec"),
        ("bug_report_adaptive_sentence_False_sec.json", "bug_report_precise_sentence_True.json", "adaptive_sentence_False_sec"),
        ("bug_report_balance_sentence_True_one.json", "bug_report_precise_sentence_True.json", "balance_sentence_True_one"),
        ("bug_report_balance_sentence_True_sec.json", "bug_report_precise_sentence_True.json", "balance_sentence_True_sec"),
        ("bug_report_balance_sentence_False_sec.json", "bug_report_precise_sentence_True.json", "balance_sentence_False_sec"),
    ]
    
    results = []
    
    for adaptive_file, precise_file, description in datasets:
        try:
            # Load data with folder path
            adaptive_data = load_json_data(bug_reports_folder + adaptive_file)
            precise_data = load_json_data(bug_reports_folder + precise_file)
            
            # Analyze
            adaptive_count, precise_count, contained_count, not_contained_count, other_timing, precise_timing = analyze_datasets(
                adaptive_data, precise_data
            )
            
            # Store results
            results.append((description, adaptive_count, precise_count, contained_count, not_contained_count, other_timing, precise_timing))
            
            # Print detailed results
            print_analysis(description, adaptive_count, precise_count, contained_count, not_contained_count, other_timing, precise_timing)
            
        except Exception as e:
            print(f"Error analyzing {description}: {e}")
            continue
    
    # Print summary table
    print_summary(results)

if __name__ == "__main__":
    main() 