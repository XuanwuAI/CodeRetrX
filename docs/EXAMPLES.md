# Examples Guide

This guide provides practical examples of using CodeLib for various code analysis tasks.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Static Analysis Examples](#static-analysis-examples)
- [LLM-Powered Analysis](#llm-powered-analysis)
- [Advanced Workflows](#advanced-workflows)
- [Security Analysis](#security-analysis)
- [Code Quality Assessment](#code-quality-assessment)
- [Integration Examples](#integration-examples)

## Basic Usage

### Loading and Exploring a Codebase

```python
from codelib import Codebase

# Load a codebase
codebase = Codebase.from_path("/path/to/your/project")

# Basic statistics
print(f"Total files: {len(codebase.files)}")
print(f"Total symbols: {len(codebase.symbols)}")
print(f"Languages found: {set(f.language for f in codebase.files)}")

# File breakdown by type
from collections import Counter
file_types = Counter(f.type for f in codebase.files)
print("File types:", dict(file_types))
```

### Simple Pattern Search

```python
# Search for TODO comments
todos = codebase.search(r"TODO|FIXME|XXX")
print(f"Found {len(todos)} TODO items")

for todo in todos[:5]:  # Show first 5
    print(f"{todo.file_path}:{todo.line_number} - {todo.line_content.strip()}")
```

## Static Analysis Examples

### Finding Functions and Classes

```python
from codelib.static import ripgrep_search

# Find all async functions
async_functions = ripgrep_search(
    "/path/to/project",
    r"async\s+def\s+(\w+)",
    file_types=["*.py"]
)

# Find all class definitions
classes = ripgrep_search(
    "/path/to/project",
    r"class\s+(\w+)",
    file_types=["*.py"]
)

print(f"Found {len(async_functions)} async functions")
print(f"Found {len(classes)} classes")
```

### Analyzing Import Patterns

```python
# Find import statements
imports = codebase.search(r"^(import|from)\s+", file_types=["*.py"])

# Categorize imports
standard_lib = []
third_party = []
local_imports = []

for imp in imports:
    content = imp.line_content.strip()
    if content.startswith("from .") or content.startswith("import ."):
        local_imports.append(imp)
    elif any(lib in content for lib in ["os", "sys", "json", "re", "datetime"]):
        standard_lib.append(imp)
    else:
        third_party.append(imp)

print(f"Standard library imports: {len(standard_lib)}")
print(f"Third-party imports: {len(third_party)}")
print(f"Local imports: {len(local_imports)}")
```

### Code Complexity Analysis

```python
# Find deeply nested code (potential complexity issues)
nested_code = codebase.search(r"^\s{12,}", file_types=["*.py"])

# Find long functions (basic heuristic)
long_functions = []
for file in codebase.files:
    if file.language == "python":
        lines = file.content.split('\n')
        in_function = False
        func_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if in_function:
                    func_length = i - func_start
                    if func_length > 50:  # Functions longer than 50 lines
                        long_functions.append((file.path, func_start, func_length))
                in_function = True
                func_start = i

print(f"Found {len(nested_code)} deeply nested code blocks")
print(f"Found {len(long_functions)} long functions")
```

## LLM-Powered Analysis

### Semantic Code Search

```python
from codelib.retrieval import SmartCodebase

# Initialize with LLM capabilities
smart_codebase = SmartCodebase.from_path("/path/to/project")

# Search for authentication-related code
auth_code = smart_codebase.search(
    "functions and classes that handle user authentication and authorization",
    strategy="semantic"
)

print(f"Found {len(auth_code)} authentication-related code elements")
for element in auth_code[:3]:
    print(f"- {element.name} in {element.file_path}")
```

### Topic Extraction

```python
# Extract main topics from the codebase
topics = smart_codebase.extract_topics(limit=10)
print("Main topics in the codebase:")
for i, topic in enumerate(topics, 1):
    print(f"{i}. {topic}")

# Find code related to specific topics
database_code = smart_codebase.search(
    "database operations, SQL queries, and data persistence",
    strategy="semantic"
)
```

### Intelligent Code Filtering

```python
from codelib.retrieval import multi_strategy_code_filter

# First, get all function definitions
all_functions = codebase.search(r"def\s+(\w+)", file_types=["*.py"])

# Filter for API-related functions
api_functions = multi_strategy_code_filter(
    all_functions,
    "functions that handle HTTP requests, API endpoints, or web service calls"
)

print(f"Filtered from {len(all_functions)} to {len(api_functions)} API functions")
```

## Advanced Workflows

### Multi-Language Analysis

```python
def analyze_multilang_project(project_path):
    """Analyze a project with multiple programming languages."""
    codebase = Codebase.from_path(project_path)
    
    # Group files by language
    by_language = {}
    for file in codebase.files:
        lang = file.language
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(file)
    
    # Analyze each language
    results = {}
    for lang, files in by_language.items():
        if lang == "python":
            results[lang] = analyze_python_files(files)
        elif lang == "javascript":
            results[lang] = analyze_javascript_files(files)
        elif lang == "java":
            results[lang] = analyze_java_files(files)
    
    return results

def analyze_python_files(files):
    """Specific analysis for Python files."""
    total_lines = sum(len(f.content.split('\n')) for f in files)
    
    # Find decorators
    decorators = []
    for file in files:
        lines = file.content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('@'):
                decorators.append((file.path, i+1, line.strip()))
    
    return {
        "total_lines": total_lines,
        "decorators": len(decorators),
        "files": len(files)
    }
```

### Code Evolution Analysis

```python
from codelib.utils import GitHistoryAnalyzer

def analyze_code_evolution(repo_path):
    """Analyze how code has evolved over time."""
    codebase = Codebase.from_path(repo_path)
    
    # Get git history
    history = GitHistoryAnalyzer(repo_path)
    
    # Analyze changes over time
    evolution_data = {}
    for commit in history.get_commits(limit=50):
        snapshot = history.get_codebase_at_commit(commit.hash)
        evolution_data[commit.date] = {
            "files": len(snapshot.files),
            "lines": sum(len(f.content.split('\n')) for f in snapshot.files),
            "symbols": len(snapshot.symbols)
        }
    
    return evolution_data
```

## Security Analysis

### Finding Security Vulnerabilities

```python
def security_audit(codebase):
    """Perform a basic security audit of the codebase."""
    security_issues = []
    
    # Check for dangerous function calls
    dangerous_patterns = [
        (r"eval\(", "Dangerous eval() usage"),
        (r"exec\(", "Dangerous exec() usage"),
        (r"os\.system\(", "System command execution"),
        (r"subprocess\.call\(", "Subprocess call"),
        (r"pickle\.loads\(", "Unsafe pickle deserialization"),
        (r"yaml\.load\(", "Unsafe YAML loading"),
        (r"shell=True", "Shell=True in subprocess"),
    ]
    
    for pattern, description in dangerous_patterns:
        matches = codebase.search(pattern, file_types=["*.py"])
        for match in matches:
            security_issues.append({
                "type": "dangerous_function",
                "description": description,
                "file": match.file_path,
                "line": match.line_number,
                "code": match.line_content.strip()
            })
    
    # Check for hardcoded secrets
    secret_patterns = [
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
        (r"token\s*=\s*['\"][^'\"]+['\"]", "Hardcoded token"),
    ]
    
    for pattern, description in secret_patterns:
        matches = codebase.search(pattern, ignore_case=True)
        for match in matches:
            security_issues.append({
                "type": "hardcoded_secret",
                "description": description,
                "file": match.file_path,
                "line": match.line_number,
                "code": match.line_content.strip()
            })
    
    return security_issues

# Run security audit
issues = security_audit(codebase)
print(f"Found {len(issues)} potential security issues")

# Group by type
from collections import defaultdict
by_type = defaultdict(list)
for issue in issues:
    by_type[issue["type"]].append(issue)

for issue_type, issue_list in by_type.items():
    print(f"\n{issue_type}: {len(issue_list)} issues")
    for issue in issue_list[:3]:  # Show first 3
        print(f"  - {issue['description']} in {issue['file']}:{issue['line']}")
```

### LLM-Enhanced Security Analysis

```python
from codelib.retrieval import SmartCodebase

def advanced_security_analysis(project_path):
    """Use LLM to identify potential security issues."""
    smart_codebase = SmartCodebase.from_path(project_path)
    
    # Find authentication/authorization code
    auth_code = smart_codebase.search(
        "authentication, authorization, access control, and security validation code",
        strategy="semantic"
    )
    
    # Find input validation
    validation_code = smart_codebase.search(
        "input validation, data sanitization, and parameter checking",
        strategy="semantic"
    )
    
    # Find encryption/cryptography
    crypto_code = smart_codebase.search(
        "encryption, decryption, hashing, and cryptographic operations",
        strategy="semantic"
    )
    
    return {
        "authentication": len(auth_code),
        "validation": len(validation_code),
        "cryptography": len(crypto_code),
        "details": {
            "auth_files": [c.file_path for c in auth_code],
            "validation_files": [c.file_path for c in validation_code],
            "crypto_files": [c.file_path for c in crypto_code]
        }
    }
```

## Code Quality Assessment

### Maintainability Analysis

```python
def assess_maintainability(codebase):
    """Assess code maintainability."""
    metrics = {}
    
    # Documentation coverage
    python_files = [f for f in codebase.files if f.language == "python"]
    documented_functions = 0
    total_functions = 0
    
    for file in python_files:
        lines = file.content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                total_functions += 1
                # Check if next few lines contain docstring
                for j in range(i+1, min(i+5, len(lines))):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        documented_functions += 1
                        break
    
    doc_coverage = documented_functions / total_functions if total_functions > 0 else 0
    
    # Test coverage estimation
    test_files = [f for f in codebase.files if "test" in f.path.lower()]
    test_coverage_estimate = len(test_files) / len(python_files) if python_files else 0
    
    # Complexity indicators
    complex_conditions = codebase.search(r"if.*and.*or", file_types=["*.py"])
    long_parameter_lists = codebase.search(r"def\s+\w+\([^)]{50,}\)", file_types=["*.py"])
    
    return {
        "documentation_coverage": doc_coverage,
        "test_coverage_estimate": test_coverage_estimate,
        "complex_conditions": len(complex_conditions),
        "long_parameter_lists": len(long_parameter_lists),
        "maintainability_score": calculate_maintainability_score(
            doc_coverage, test_coverage_estimate, 
            len(complex_conditions), len(long_parameter_lists)
        )
    }

def calculate_maintainability_score(doc_cov, test_cov, complex_cond, long_params):
    """Calculate a simple maintainability score (0-100)."""
    score = 50  # Base score
    score += doc_cov * 25  # Documentation adds up to 25 points
    score += test_cov * 25  # Tests add up to 25 points
    score -= min(complex_cond, 25)  # Complex conditions reduce score
    score -= min(long_params, 25)  # Long parameter lists reduce score
    return max(0, min(100, score))
```

### Performance Analysis

```python
def analyze_performance_patterns(codebase):
    """Identify potential performance issues."""
    performance_issues = []
    
    # Find nested loops
    nested_loops = codebase.search(r"for.*:\s*\n\s*for", file_types=["*.py"])
    
    # Find database queries in loops
    db_in_loops = codebase.search(r"for.*:\s*.*\.execute\(", file_types=["*.py"])
    
    # Find inefficient string concatenation
    string_concat = codebase.search(r"\+\s*=\s*.*['\"]", file_types=["*.py"])
    
    # Find large data structures
    large_lists = codebase.search(r"range\(\s*\d{4,}\s*\)", file_types=["*.py"])
    
    return {
        "nested_loops": len(nested_loops),
        "db_queries_in_loops": len(db_in_loops),
        "string_concatenation": len(string_concat),
        "large_ranges": len(large_lists),
        "issues": performance_issues
    }
```

## Integration Examples

### Continuous Integration Integration

```python
#!/usr/bin/env python3
"""
CI/CD integration script for code analysis.
"""
import sys
import json
from codelib import Codebase
from codelib.retrieval import SmartCodebase

def ci_analysis(project_path, output_file="analysis_report.json"):
    """Run analysis suitable for CI/CD pipeline."""
    
    # Basic static analysis
    codebase = Codebase.from_path(project_path)
    
    # Security check
    security_issues = security_audit(codebase)
    
    # Quality metrics
    quality_metrics = assess_maintainability(codebase)
    
    # Performance analysis
    performance_metrics = analyze_performance_patterns(codebase)
    
    # Generate report
    report = {
        "summary": {
            "total_files": len(codebase.files),
            "total_lines": sum(len(f.content.split('\n')) for f in codebase.files),
            "languages": list(set(f.language for f in codebase.files))
        },
        "security": {
            "issues_found": len(security_issues),
            "critical_issues": len([i for i in security_issues if i["type"] == "dangerous_function"])
        },
        "quality": quality_metrics,
        "performance": performance_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Exit with error code if critical issues found
    if report["security"]["critical_issues"] > 0:
        print(f"CRITICAL: Found {report['security']['critical_issues']} critical security issues")
        sys.exit(1)
    
    print(f"Analysis complete. Report saved to {output_file}")
    return report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ci_analysis.py <project_path>")
        sys.exit(1)
    
    ci_analysis(sys.argv[1])
```

### IDE Plugin Integration

```python
class CodeAnalysisPlugin:
    """Example IDE plugin integration."""
    
    def __init__(self, workspace_path):
        self.codebase = Codebase.from_path(workspace_path)
        self.smart_codebase = SmartCodebase.from_path(workspace_path)
    
    def analyze_current_file(self, file_path):
        """Analyze the currently open file."""
        # Find the file in the codebase
        current_file = next(
            (f for f in self.codebase.files if f.path == file_path), 
            None
        )
        
        if not current_file:
            return {"error": "File not found in codebase"}
        
        # Analyze the file
        analysis = {
            "file_info": {
                "path": current_file.path,
                "language": current_file.language,
                "size": current_file.size,
                "lines": len(current_file.content.split('\n'))
            },
            "symbols": [
                {"name": s.name, "type": s.type, "line": s.line_number}
                for s in current_file.symbols
            ],
            "issues": self.find_issues_in_file(current_file)
        }
        
        return analysis
    
    def find_issues_in_file(self, file):
        """Find potential issues in a specific file."""
        issues = []
        
        # Check for TODO comments
        todos = codebase.search(r"TODO|FIXME|XXX", files=[file.path])
        issues.extend([
            {"type": "todo", "line": t.line_number, "message": t.line_content.strip()}
            for t in todos
        ])
        
        # Check for long lines
        lines = file.content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100:
                issues.append({
                    "type": "long_line",
                    "line": i + 1,
                    "message": f"Line too long ({len(line)} characters)"
                })
        
        return issues
    
    def get_suggestions(self, query):
        """Get code suggestions based on query."""
        results = self.smart_codebase.search(query, strategy="semantic")
        
        return [
            {
                "title": r.name,
                "file": r.file_path,
                "line": r.line_number,
                "preview": r.content[:100] + "..." if len(r.content) > 100 else r.content
            }
            for r in results[:10]
        ]
```

### Custom Analysis Pipeline

```python
from typing import List, Dict, Any
import asyncio

class AnalysisPipeline:
    """Custom analysis pipeline for complex workflows."""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.codebase = Codebase.from_path(project_path)
        self.smart_codebase = SmartCodebase.from_path(project_path)
        
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        
        # Run analyses in parallel
        tasks = [
            self.static_analysis(),
            self.security_analysis(),
            self.quality_analysis(),
            self.semantic_analysis(),
            self.dependency_analysis()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        return {
            "static": results[0],
            "security": results[1],
            "quality": results[2],
            "semantic": results[3],
            "dependencies": results[4],
            "summary": self.generate_summary(results)
        }
    
    async def static_analysis(self):
        """Run static analysis."""
        return {
            "total_files": len(self.codebase.files),
            "languages": list(set(f.language for f in self.codebase.files)),
            "symbols": len(self.codebase.symbols),
            "complexity_metrics": self.calculate_complexity()
        }
    
    async def security_analysis(self):
        """Run security analysis."""
        return security_audit(self.codebase)
    
    async def quality_analysis(self):
        """Run quality analysis."""
        return assess_maintainability(self.codebase)
    
    async def semantic_analysis(self):
        """Run semantic analysis using LLM."""
        topics = self.smart_codebase.extract_topics(limit=10)
        return {
            "topics": topics,
            "architecture_components": await self.identify_architecture()
        }
    
    async def dependency_analysis(self):
        """Analyze dependencies."""
        deps = self.codebase.get_dependencies()
        return {
            "total_dependencies": len(deps),
            "external_dependencies": len([d for d in deps if d.is_external]),
            "dependency_tree": self.build_dependency_tree(deps)
        }
    
    def generate_summary(self, results):
        """Generate executive summary."""
        return {
            "overall_health": self.calculate_health_score(results),
            "key_findings": self.extract_key_findings(results),
            "recommendations": self.generate_recommendations(results)
        }

# Usage
async def main():
    pipeline = AnalysisPipeline("/path/to/project")
    results = await pipeline.run_full_analysis()
    
    print("Analysis complete!")
    print(f"Overall health score: {results['summary']['overall_health']}")
    print("Key findings:")
    for finding in results['summary']['key_findings']:
        print(f"  - {finding}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Memory Management for Large Codebases

```python
from codelib import Codebase
import gc

def analyze_large_codebase(project_path):
    """Efficiently analyze large codebases."""
    
    # Use generator for file processing
    def process_files_in_batches(files, batch_size=100):
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            yield batch
    
    codebase = Codebase.from_path(project_path)
    results = []
    
    # Process files in batches
    for batch in process_files_in_batches(codebase.files):
        batch_results = analyze_file_batch(batch)
        results.extend(batch_results)
        
        # Force garbage collection
        gc.collect()
    
    return results

def analyze_file_batch(files):
    """Analyze a batch of files."""
    # Process files efficiently
    return [{"file": f.path, "lines": len(f.content.split('\n'))} for f in files]
```

### Error Handling and Logging

```python
import logging
from codelib.exceptions import CodebaseError, LLMError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_analysis(project_path):
    """Analysis with proper error handling."""
    
    try:
        codebase = Codebase.from_path(project_path)
        logger.info(f"Loaded codebase with {len(codebase.files)} files")
        
        # Perform analysis with error handling
        results = {}
        
        try:
            results["static"] = perform_static_analysis(codebase)
        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
            results["static"] = {"error": str(e)}
        
        try:
            results["llm"] = perform_llm_analysis(codebase)
        except LLMError as e:
            logger.error(f"LLM analysis failed: {e}")
            results["llm"] = {"error": "LLM service unavailable"}
        
        return results
        
    except CodebaseError as e:
        logger.error(f"Failed to load codebase: {e}")
        return {"error": f"Codebase loading failed: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}
```

This examples guide demonstrates the versatility and power of CodeLib for various code analysis tasks. From basic pattern matching to advanced LLM-powered semantic analysis, these examples show how to leverage the library's capabilities for real-world scenarios.