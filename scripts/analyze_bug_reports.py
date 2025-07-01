import json
import sys
from pathlib import Path
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

def extract_input_tokens(data: Dict) -> Optional[int]:
    """Extract input tokens from bug report data"""
    return data.get('input_tokens')

def extract_output_tokens(data: Dict) -> Optional[int]:
    """Extract output tokens from bug report data"""
    return data.get('output_tokens')

def extract_llm_cost(data: Dict) -> Optional[float]:
    """Extract LLM cost from bug report data"""
    return data.get('llm_cost')

def analyze_datasets(other_data: Dict, precise_data: Dict) -> Tuple[int, int, int, int, Optional[int], Optional[int], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Analyze the relationship between different methods.
    
    Returns:
        - fast_method_count: Total number of items in fast/balance/adaptive dataset
        - ground_truth_count: Total number of items in precise dataset
        - contained_count: Number of fast/balance/adaptive items contained in precise dataset
        - not_contained_count: Number of fast/balance/adaptive items not contained in precise dataset
        - other_input_tokens: Input tokens from other data
        - other_output_tokens: Output tokens from other data
        - precise_input_tokens: Input tokens from precise data
        - precise_output_tokens: Output tokens from precise data
        - other_cost: LLM cost from other data
        - precise_cost: LLM cost from precise data
    """
    method_ids = extract_bug_ids(other_data)
    precise_ids = extract_bug_ids(precise_data)
    
    # Remove empty IDs
    method_ids.discard('')
    precise_ids.discard('')
    
    # Calculate intersections
    contained_ids = method_ids.intersection(precise_ids)
    not_contained_ids = method_ids - precise_ids
    
    # Extract token and cost information
    other_input_tokens = extract_input_tokens(other_data)
    other_output_tokens = extract_output_tokens(other_data)
    precise_input_tokens = extract_input_tokens(precise_data)
    precise_output_tokens = extract_output_tokens(precise_data)
    other_cost = extract_llm_cost(other_data)
    precise_cost = extract_llm_cost(precise_data)
    
    return len(method_ids), len(precise_ids), len(contained_ids), len(not_contained_ids), other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost

def print_token_comparison(other_input_tokens: Optional[int], other_output_tokens: Optional[int], 
                          precise_input_tokens: Optional[int], precise_output_tokens: Optional[int], 
                          other_cost: Optional[float], precise_cost: Optional[float], method_name: str):
    print(f"\n--- Token & Cost Comparison for {method_name} ---")
    
    # Token comparison
    print(f"\n--- Token Usage Comparison ---")
    if other_input_tokens is not None and precise_input_tokens is not None:
        print(f"Input tokens - Other: {other_input_tokens:,}, Precise: {precise_input_tokens:,}")
        if other_input_tokens > 0:
            input_ratio = precise_input_tokens / other_input_tokens
            print(f"Input token ratio (Precise/Other): {input_ratio:.2f}x")
    elif other_input_tokens is not None:
        print(f"Other method input tokens: {other_input_tokens:,}")
        print("Precise input token data not available")
    elif precise_input_tokens is not None:
        print(f"Precise input tokens: {precise_input_tokens:,}")
        print("Other method input token data not available")
    else:
        print("No input token data available for comparison")
    
    if other_output_tokens is not None and precise_output_tokens is not None:
        print(f"Output tokens - Other: {other_output_tokens:,}, Precise: {precise_output_tokens:,}")
        if other_output_tokens > 0:
            output_ratio = precise_output_tokens / other_output_tokens
            print(f"Output token ratio (Precise/Other): {output_ratio:.2f}x")
    elif other_output_tokens is not None:
        print(f"Other method output tokens: {other_output_tokens:,}")
        print("Precise output token data not available")
    elif precise_output_tokens is not None:
        print(f"Precise output tokens: {precise_output_tokens:,}")
        print("Other method output token data not available")
    else:
        print("No output token data available for comparison")
    
    # Total tokens comparison
    if (other_input_tokens is not None and other_output_tokens is not None and 
        precise_input_tokens is not None and precise_output_tokens is not None):
        other_total_tokens = other_input_tokens + other_output_tokens
        precise_total_tokens = precise_input_tokens + precise_output_tokens
        print(f"Total tokens - Other: {other_total_tokens:,}, Precise: {precise_total_tokens:,}")
        if other_total_tokens > 0:
            total_ratio = precise_total_tokens / other_total_tokens
            print(f"Total token ratio (Precise/Other): {total_ratio:.2f}x")
    
    # Cost comparison
    print(f"\n--- LLM Cost Comparison ---")
    if other_cost is not None and precise_cost is not None:
        print(f"Other method LLM cost: ${other_cost:.6f}")
        print(f"Precise LLM cost: ${precise_cost:.6f}")
        
        if other_cost > 0:
            cost_ratio = precise_cost / other_cost
            print(f"Cost ratio (Precise/Other): {cost_ratio:.2f}x")
            
        cost_savings = precise_cost - other_cost
        if cost_savings != 0:
            cost_savings_pct = (cost_savings / precise_cost * 100) if precise_cost > 0 else 0
            if cost_savings > 0:
                print(f"Cost increase: ${cost_savings:.6f} (+{cost_savings_pct:.1f}%)")
            else:
                print(f"Cost savings: ${-cost_savings:.6f} ({-cost_savings_pct:.1f}%)")
    elif other_cost is not None:
        print(f"Other method LLM cost: ${other_cost:.6f}")
        print("Precise cost data not available")
    elif precise_cost is not None:
        print(f"Precise LLM cost: ${precise_cost:.6f}")
        print("Other method cost data not available")
    else:
        print("No cost data available for comparison")

def print_analysis(filename_prefix: str, method_count: int, precise_count: int, 
                  contained_count: int, not_contained_count: int, 
                  other_input_tokens: Optional[int] = None, other_output_tokens: Optional[int] = None,
                  precise_input_tokens: Optional[int] = None, precise_output_tokens: Optional[int] = None,
                  other_cost: Optional[float] = None, precise_cost: Optional[float] = None):
    print(f"\n=== Analysis for {filename_prefix} ===")
    print(f"Total other method items: {method_count}")
    print(f"Total precise items: {precise_count}")
    print(f"Other method items contained in precise: {contained_count}")
    print(f"Other method items NOT contained in precise: {not_contained_count}")    
    
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
    
    # Add token and cost comparison
    print_token_comparison(other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost, filename_prefix)
    
    print("-" * 60)

def print_summary(results: List[Tuple[str, int, int, int, int, Optional[int], Optional[int], Optional[int], Optional[int], Optional[float], Optional[float]]]):
    print("\n" + "="*155)
    print("SUMMARY TABLE")
    print("="*155)
    
    headers = ["Dataset", "Other", "Precise", "Contained", "Coverage %", "In Tokens (O)", "In Tokens (P)", "Out Tokens (O)", "Out Tokens (P)", "Cost (O)", "Cost (P)", "Cost Ratio"]
    print(f"{headers[0]:<30} {headers[1]:<9} {headers[2]:<8} {headers[3]:<10} {headers[4]:<11} {headers[5]:<12} {headers[6]:<12} {headers[7]:<13} {headers[8]:<13} {headers[9]:<10} {headers[10]:<10} {headers[11]:<10}")
    print("-" * 155)
    
    for desc, method_count, precise_count, contained_count, not_contained_count, other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost in results:
        coverage_pct = (contained_count / precise_count * 100) if precise_count > 0 else 0
        
        # Format token strings
        other_input_str = f"{other_input_tokens:,}" if other_input_tokens is not None else "N/A"
        precise_input_str = f"{precise_input_tokens:,}" if precise_input_tokens is not None else "N/A"
        other_output_str = f"{other_output_tokens:,}" if other_output_tokens is not None else "N/A"
        precise_output_str = f"{precise_output_tokens:,}" if precise_output_tokens is not None else "N/A"
        
        # Format cost strings
        other_cost_str = f"${other_cost:.4f}" if other_cost is not None else "N/A"
        precise_cost_str = f"${precise_cost:.4f}" if precise_cost is not None else "N/A"
        
        cost_ratio = precise_cost / other_cost if other_cost and other_cost > 0 and precise_cost is not None else 0
        cost_ratio_str = f"{cost_ratio:.2f}x" if cost_ratio > 0 else "N/A"
        
        print(f"{desc:<30} {method_count:<9} {precise_count:<8} {contained_count:<10} {coverage_pct:>9.1f}% {other_input_str:>11} {precise_input_str:>11} {other_output_str:>12} {precise_output_str:>12} {other_cost_str:>9} {precise_cost_str:>9} {cost_ratio_str:>9}")



def main():
    
    bug_reports_folder = Path("bug_reports")
    results = []
    
    # Find all repo directories
    for repo_dir in bug_reports_folder.iterdir():
        if not repo_dir.is_dir():
            continue
            
        print(f"\nAnalyzing repository: {repo_dir.name}")
        
        # Find precise file
        precise_files = list(repo_dir.glob("*_precise_*.json"))
        if not precise_files:
            print(f"No precise file found for {repo_dir.name}")
            continue
        precise_file = precise_files[0]
        
        # Find all non-precise files
        other_files = [f for f in repo_dir.glob("*.json") if "_precise_" not in f.name]
        
        for other_file in other_files:
            try:
                # Extract mode from filename
                mode = "unknown"
                for m in ["fast", "balance", "adaptive"]:
                    if f"_{m}_" in other_file.name:
                        mode = m
                        break
                
                description = f"{repo_dir.name}_{mode}"
                
                # Load data
                other_data = load_json_data(str(other_file))
                precise_data = load_json_data(str(precise_file))
                
                # Analyze
                other_count, precise_count, contained_count, not_contained_count, other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost = analyze_datasets(
                    other_data, precise_data
                )
                
                # Store results
                results.append((description, other_count, precise_count, contained_count, not_contained_count, other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost))
                
                # Print detailed results
                print_analysis(description, other_count, precise_count, contained_count, not_contained_count, other_input_tokens, other_output_tokens, precise_input_tokens, precise_output_tokens, other_cost, precise_cost)
                
            except Exception as e:
                print(f"Error analyzing {other_file.name}: {e}")
                continue
    
    # Print summary table
    if results:
        print_summary(results)
    else:
        print("No valid comparisons found")

if __name__ == "__main__":
    main() 