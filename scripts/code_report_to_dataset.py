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
    """Load a JSON file and return its content.

    Be tolerant of empty files or invalid JSON by returning an empty dict.
    """
    print(f"Loading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if not content.strip():
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}, starting from empty dataset.")
            return {}

def derive_repo_key_from_url(repo_url: str) -> str:
    """Derive a stable repo key like 'owner_repo' from a repo URL or path string."""
    if not repo_url:
        return "unknown_repo"
    # Strip possible .git suffix and trailing slashes
    url = repo_url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    # Split by '/' and take last two components if possible
    parts = url.split('/')
    if len(parts) >= 2:
        owner = parts[-2] or "unknown"
        repo = parts[-1] or "repo"
        return f"{owner}_{repo}"
    # Fallback to replacing slashes
    return url.replace('/', '_')

def normalize_dataset_shape(dataset: Any) -> Dict[str, Any]:
    """
    Normalize dataset into a multi-repo mapping shape:
    {
      "owner_repo": { "repo_url": str, "commit": str, "queries": [...] },
      ...
    }

    Accepts previous single-repo shape or already-normalized mapping or empty.
    """
    if not dataset:
        return {}
    # Already mapping of repo_key -> repo_obj
    if isinstance(dataset, dict) and 'queries' not in dataset:
        # Heuristic: if keys do not look like dataset fields, assume repo map
        # but still ensure each value has expected fields
        return dataset
    # Single repo shape
    if isinstance(dataset, dict) and 'repo_url' in dataset and 'queries' in dataset:
        repo_key = derive_repo_key_from_url(dataset.get('repo_url') or '')
        return {repo_key: dataset}
    # Unexpected types (e.g., list); convert conservatively
    if isinstance(dataset, list):
        # Try to merge any items that look like single-repo entries
        repo_map: Dict[str, Any] = {}
        for item in dataset:
            if isinstance(item, dict) and 'repo_url' in item and 'queries' in item:
                key = derive_repo_key_from_url(item.get('repo_url') or '')
                repo_map[key] = item
        return repo_map
    return {}

def ensure_repo_entry(dataset_map: Dict[str, Any], repo_url: str, commit: str = None) -> Dict[str, Any]:
    """Ensure a repo entry exists in dataset_map and return it."""
    repo_key = derive_repo_key_from_url(repo_url)
    if repo_key not in dataset_map:
        dataset_map[repo_key] = {
            'repo_url': repo_url,
            'commit': commit or 'unknown',
            'queries': []
        }
    else:
        # Populate missing fields if absent
        repo_obj = dataset_map[repo_key]
        if 'repo_url' not in repo_obj:
            repo_obj['repo_url'] = repo_url
        if 'commit' not in repo_obj:
            repo_obj['commit'] = commit or repo_obj.get('commit') or 'unknown'
        if 'queries' not in repo_obj or not isinstance(repo_obj['queries'], list):
            repo_obj['queries'] = []
    return dataset_map[repo_key]

def build_prompt_cost_index(prompt_costs: List[Dict[str, Any]]):
    """Build an index for best-prefix cost lookup supporting truncated prompts ending with '...'"""
    prefixes: List[Dict[str, Any]] = []
    for pc in prompt_costs or []:
        prompt = pc.get('prompt') or ''
        cost = pc.get('cost', 0.0)
        if prompt.endswith('...'):
            prefix = prompt[:-3].rstrip()
        else:
            prefix = prompt.rstrip()
        if prefix:
            prefixes.append({'prefix': prefix, 'cost': cost, 'length': len(prefix)})
    # Sort by descending length to prefer the longest match
    prefixes.sort(key=lambda x: x['length'], reverse=True)
    return prefixes

def lookup_cost_for_code_type(code_type: str, prefix_index: List[Dict[str, Any]], fallback: float = 0.0) -> float:
    if not code_type:
        return fallback
    for entry in prefix_index:
        if code_type.startswith(entry['prefix']):
            return entry['cost']
    return fallback

def find_matching_query(dataset_queries: List[Dict], code_type: str) -> Dict[str, Any]:
    """
    Find the query in dataset that matches the code_type from code_report.
    This is done by comparing the filter_prompt with the code_type.
    """
    for query in dataset_queries:
        if query.get('filter_prompt') == code_type:
            return query
    return None

def derive_queries_file_from_dataset(dataset_path: str) -> str:
    """Given a dataset path like bench/dataset_PQAlgo.json, derive bench/queries_PQAlgo.json."""
    base_dir = os.path.dirname(dataset_path)
    base_name = os.path.basename(dataset_path)
    if base_name.startswith('dataset_'):
        suffix = base_name[len('dataset_'):]
        return os.path.join(base_dir, f"queries_{suffix}")
    # Fallback: attempt queries.json next to dataset
    return os.path.join(base_dir, 'queries.json')

def load_standard_query_name_map(dataset_path: str) -> Dict[str, str]:
    """Load a mapping from filter_prompt -> canonical name from the derived queries file.

    If no queries file exists, return an empty mapping.
    """
    mapping: Dict[str, str] = {}
    queries_file = derive_queries_file_from_dataset(dataset_path)
    if os.path.exists(queries_file):
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            if isinstance(queries, list):
                for q in queries:
                    fp = q.get('filter_prompt')
                    nm = q.get('name')
                    if fp and nm:
                        mapping[fp] = nm
        except Exception:
            # Be tolerant: if anything goes wrong, just return empty mapping
            return {}
    return mapping

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
    dataset_raw = load_json_file(dataset_path)
    
    print(f"Found {len(code_report.get('codes', []))} codes in code_report")
    # Normalize dataset to multi-repo mapping shape
    dataset = normalize_dataset_shape(dataset_raw)
    print(f"Found {len(dataset.keys())} repos in dataset")
    
    # Extract cost information from code_report
    cost_info = code_report.get('cost_info', {})
    prompt_costs = cost_info.get('prompt_costs', [])
    prefix_index = build_prompt_cost_index(prompt_costs)
    # Load canonical names for queries (filter_prompt -> name)
    canonical_name_map = load_standard_query_name_map(dataset_path)
    
    # Prepare queries list from queries_vuln file (pivot by queries, not codes)
    queries_file = derive_queries_file_from_dataset(dataset_path)
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_list = json.load(f)
            if not isinstance(queries_list, list):
                queries_list = []
    except Exception:
        queries_list = []

    # Counters
    merged_count = 0
    skipped_count = 0
    
    # Determine repo information from code_report
    repo_url = code_report.get('repository') or code_report.get('repo_url') or ''
    commit = code_report.get('commit') or 'unknown'
    repo_obj = ensure_repo_entry(dataset, repo_url=repo_url, commit=commit)

    # Iterate according to queries_vuln, not codes
    for q in queries_list:
        filter_prompt = q.get('filter_prompt')
        if not filter_prompt:
            continue

        # Find matching query in the repo (create one if missing)
        matching_query = find_matching_query(repo_obj.get('queries', []), filter_prompt)
        if not matching_query:
            canonical_name = canonical_name_map.get(filter_prompt) or q.get('name') or filter_prompt
            matching_query = {
                'name': canonical_name,
                'filter_prompt': filter_prompt,
                'subdirs_or_files': q.get('subdirs_or_files') or ['/'],
                'snippets': [],
                'cost': 0.0
            }
            repo_obj['queries'].append(matching_query)
        else:
            canonical_name = canonical_name_map.get(filter_prompt) or q.get('name')
            if canonical_name and matching_query.get('name') != canonical_name:
                matching_query['name'] = canonical_name
            if 'subdirs_or_files' not in matching_query:
                matching_query['subdirs_or_files'] = q.get('subdirs_or_files') or ['/']

        # Collect all codes that correspond to this query's filter_prompt
        related_codes = [c for c in code_report.get('codes', []) if c.get('code_type') == filter_prompt]

        # Aggregate all locations into snippets
        snippets: List[Dict[str, Any]] = []
        for code in related_codes:
            for location in code.get('locations', []) or []:
                start_line = location.get('start_line')
                end_line = location.get('end_line')
                try:
                    if start_line is not None:
                        start_line = int(start_line)
                    if end_line is not None:
                        end_line = int(end_line)
                except (ValueError, TypeError):
                    start_line = location.get('start_line')
                    end_line = location.get('end_line')
                snippet = {
                    "name": location.get('name', ''),
                    "type": location.get('type', ''),
                    "file_path": location.get('file_path', ''),
                    "start_line": start_line,
                    "end_line": end_line
                }
                if (
                    snippet['name']
                    and snippet['file_path']
                    and snippet['start_line'] is not None
                    and snippet['end_line'] is not None
                ):
                    snippets.append(snippet)

        # Add snippets, avoiding duplicates
        if 'snippets' not in matching_query:
            matching_query['snippets'] = []
        existing_snippets = matching_query['snippets']
        new_snippet_count = 0
        for snippet in snippets:
            if not any(
                existing['name'] == snippet['name'] and
                existing['file_path'] == snippet['file_path'] and
                existing['start_line'] == snippet['start_line']
                for existing in existing_snippets
            ):
                existing_snippets.append(snippet)
                new_snippet_count += 1

        # Set cost for this query based on its filter_prompt and per-prompt cost info
        cost = lookup_cost_for_code_type(filter_prompt, prefix_index, fallback=0.0)
        matching_query['cost'] = cost

        if new_snippet_count > 0:
            print(f"Added {new_snippet_count} snippets to query: {matching_query.get('name', 'Unknown')} (cost: {cost})")
            merged_count += 1
        else:
            # Still record cost even when no snippets found for this query
            print(f"No snippets for query: {matching_query.get('name', 'Unknown')} (cost: {cost})")
            skipped_count += 1
    
    # Save the merged dataset
    print(f"Saving merged dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerge completed!")
    print(f"Successfully merged: {merged_count} codes")
    print(f"Skipped: {skipped_count} codes")
    
    # Print summary per repo and per query
    print(f"\nSnippets per repo/query:")
    for repo_key, repo_obj in dataset.items():
        total_snippets = sum(len(q.get('snippets', [])) for q in repo_obj.get('queries', []))
        print(f"Repo {repo_key}: {total_snippets} snippets across {len(repo_obj.get('queries', []))} queries")
        for query in repo_obj.get('queries', []):
            snippet_count = len(query.get('snippets', []))
            if snippet_count > 0:
                print(f"  - {query.get('name', 'Unknown')}: {snippet_count} snippets")

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/code_report_to_dataset.py <code_report_path> <dataset_path> [output_path]")
        print("Example: python scripts/code_report_to_dataset.py code_reports/ollama_ollama/code_report_convert.json bench/dataset_vuln.json bench/dataset_vuln_merged.json")
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
