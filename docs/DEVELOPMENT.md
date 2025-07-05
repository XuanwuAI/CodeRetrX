# Development Guide

This guide is for developers who want to contribute to CodeLib or extend its functionality.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Extending CodeLib](#extending-codelib)
- [Performance Optimization](#performance-optimization)
- [Release Process](#release-process)

## Development Environment Setup

### Prerequisites

- Python 3.12 or higher
- Git
- uv (recommended) or pip
- Optional: Docker for containerized development

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd codelib
   ```

2. **Create Development Environment**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync --all-extras
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,stats,builtin-impl,cli]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   python -c "import codelib; print('CodeLib installed successfully')"
   pytest --version
   ```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["test/"]
}
```

#### PyCharm

1. Set project interpreter to `.venv/bin/python`
2. Configure code style to use Black
3. Enable pytest as test runner
4. Configure import sorting with isort

## Project Structure

```
codelib/
├── codelib/                 # Main library code
│   ├── __init__.py         # Public API exports
│   ├── static/             # Static analysis module
│   │   ├── __init__.py
│   │   ├── codebase/       # Codebase representation
│   │   │   ├── __init__.py
│   │   │   ├── models.py   # Data models
│   │   │   ├── loader.py   # Codebase loading logic
│   │   │   └── analyzer.py # Analysis algorithms
│   │   └── ripgrep/        # Ripgrep integration
│   │       ├── __init__.py
│   │       ├── wrapper.py  # Ripgrep wrapper
│   │       └── parser.py   # Result parsing
│   ├── retrieval/          # LLM-powered retrieval
│   │   ├── __init__.py
│   │   ├── code_recall.py  # Multi-strategy recall
│   │   ├── smart_codebase.py # Smart codebase interface
│   │   └── topic_extractor.py # Topic extraction
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration handling
│   │   ├── logging.py      # Logging utilities
│   │   └── cache.py        # Caching mechanisms
│   └── impl/               # Implementation modules
│       ├── __init__.py
│       ├── langchain/      # LangChain implementations
│       └── openai/         # OpenAI implementations
├── scripts/                # Analysis scripts
├── bench/                  # Benchmarking code
├── test/                   # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test fixtures
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # Main documentation
```

### Key Design Principles

1. **Modularity**: Each component should be independently testable
2. **Extensibility**: Use plugin architectures where appropriate
3. **Performance**: Optimize for large codebases
4. **Type Safety**: Use type hints throughout
5. **Error Handling**: Graceful degradation and informative errors

## Contributing Guidelines

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

#### Running Code Quality Checks

```bash
# Format code
black codelib/ test/

# Sort imports
isort codelib/ test/

# Lint code
flake8 codelib/ test/

# Type check
mypy codelib/
```

### Commit Message Format

Use conventional commits:

```
type(scope): short description

Longer description if needed

- Bullet points for details
- Another detail

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(retrieval): add semantic search capabilities
fix(static): handle empty files gracefully
docs(api): update SmartCodebase documentation
```

### Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Test Your Changes**
   ```bash
   pytest
   python -m pytest test/ -v
   ```

4. **Submit PR**
   - Fill out PR template
   - Link to relevant issues
   - Add screenshots/examples if applicable

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance implications considered
- [ ] Security implications reviewed

## Testing

### Test Structure

```python
# test/unit/test_codebase.py
import pytest
from codelib.static import Codebase
from codelib.exceptions import CodebaseError

class TestCodebase:
    """Test suite for Codebase class."""
    
    def test_load_from_path(self, tmp_path):
        """Test loading codebase from path."""
        # Create test files
        (tmp_path / "test.py").write_text("print('hello')")
        
        # Load codebase
        codebase = Codebase.from_path(tmp_path)
        
        # Assertions
        assert len(codebase.files) == 1
        assert codebase.files[0].path.endswith("test.py")
    
    def test_load_invalid_path(self):
        """Test loading from invalid path."""
        with pytest.raises(CodebaseError):
            Codebase.from_path("/nonexistent/path")
    
    @pytest.mark.parametrize("language,expected", [
        ("python", "py"),
        ("javascript", "js"),
        ("java", "java"),
    ])
    def test_language_detection(self, language, expected):
        """Test language detection."""
        # Implementation
        pass
```

### Test Categories

#### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution

#### Integration Tests
- Test component interactions
- Use real external services (when possible)
- Test end-to-end workflows

#### Performance Tests
- Benchmark critical operations
- Test with large datasets
- Memory usage monitoring

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest test/unit/

# Integration tests
pytest test/integration/

# With coverage
pytest --cov=codelib --cov-report=html

# Specific test file
pytest test/unit/test_codebase.py

# Specific test
pytest test/unit/test_codebase.py::TestCodebase::test_load_from_path

# Parallel execution
pytest -n auto
```

### Test Fixtures

```python
# test/conftest.py
import pytest
from pathlib import Path
from codelib.static import Codebase

@pytest.fixture
def sample_codebase(tmp_path):
    """Create a sample codebase for testing."""
    # Create sample files
    (tmp_path / "main.py").write_text("""
def hello():
    print("Hello, World!")
    
class Example:
    def method(self):
        return 42
""")
    
    (tmp_path / "utils.py").write_text("""
import os
import sys

def helper_function():
    return os.path.join("a", "b")
""")
    
    return Codebase.from_path(tmp_path)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [{"text": "Sample response"}],
        "usage": {"total_tokens": 100}
    }
```

### Testing LLM Features

```python
# test/unit/test_retrieval.py
import pytest
from unittest.mock import Mock, patch
from codelib.retrieval import SmartCodebase

class TestSmartCodebase:
    """Test LLM-powered features."""
    
    @patch('codelib.retrieval.openai.Completion.create')
    def test_semantic_search(self, mock_openai, sample_codebase):
        """Test semantic search functionality."""
        # Mock OpenAI response
        mock_openai.return_value = {
            "choices": [{"text": "def hello():"}]
        }
        
        smart_codebase = SmartCodebase(sample_codebase)
        results = smart_codebase.search("greeting functions")
        
        assert len(results) > 0
        mock_openai.assert_called_once()
    
    def test_topic_extraction(self, sample_codebase):
        """Test topic extraction."""
        smart_codebase = SmartCodebase(sample_codebase)
        
        with patch.object(smart_codebase, '_extract_topics') as mock_extract:
            mock_extract.return_value = ["utility", "example"]
            
            topics = smart_codebase.extract_topics()
            
            assert "utility" in topics
            assert "example" in topics
```

## Extending CodeLib

### Adding New Static Analysis Features

1. **Create Analysis Module**
   ```python
   # codelib/static/analyzers/complexity.py
   from typing import List, Dict
   from ..codebase import Codebase, File
   
   class ComplexityAnalyzer:
       """Analyze code complexity metrics."""
       
       def analyze_file(self, file: File) -> Dict:
           """Analyze complexity of a single file."""
           return {
               "cyclomatic_complexity": self._calculate_cyclomatic(file),
               "cognitive_complexity": self._calculate_cognitive(file),
               "nesting_depth": self._calculate_nesting(file)
           }
       
       def analyze_codebase(self, codebase: Codebase) -> Dict:
           """Analyze complexity of entire codebase."""
           results = {}
           for file in codebase.files:
               results[file.path] = self.analyze_file(file)
           return results
   ```

2. **Add to Module Init**
   ```python
   # codelib/static/__init__.py
   from .analyzers.complexity import ComplexityAnalyzer
   
   __all__ = [
       # ... existing exports
       "ComplexityAnalyzer",
   ]
   ```

3. **Add Tests**
   ```python
   # test/unit/test_complexity_analyzer.py
   import pytest
   from codelib.static.analyzers import ComplexityAnalyzer
   
   class TestComplexityAnalyzer:
       def test_analyze_simple_file(self, sample_file):
           analyzer = ComplexityAnalyzer()
           result = analyzer.analyze_file(sample_file)
           
           assert "cyclomatic_complexity" in result
           assert result["cyclomatic_complexity"] >= 1
   ```

### Adding New LLM Providers

1. **Create Provider Interface**
   ```python
   # codelib/retrieval/providers/base.py
   from abc import ABC, abstractmethod
   from typing import List, Dict, Any
   
   class LLMProvider(ABC):
       """Base class for LLM providers."""
       
       @abstractmethod
       def generate_text(self, prompt: str, **kwargs) -> str:
           """Generate text from prompt."""
           pass
       
       @abstractmethod
       def embed_text(self, text: str) -> List[float]:
           """Generate embeddings for text."""
           pass
   ```

2. **Implement Provider**
   ```python
   # codelib/retrieval/providers/anthropic.py
   from .base import LLMProvider
   import anthropic
   
   class AnthropicProvider(LLMProvider):
       """Anthropic Claude provider."""
       
       def __init__(self, api_key: str):
           self.client = anthropic.Anthropic(api_key=api_key)
       
       def generate_text(self, prompt: str, **kwargs) -> str:
           response = self.client.completions.create(
               model="claude-3-sonnet-20240229",
               prompt=prompt,
               max_tokens_to_sample=kwargs.get("max_tokens", 1000)
           )
           return response.completion
   ```

3. **Register Provider**
   ```python
   # codelib/retrieval/providers/__init__.py
   from .openai import OpenAIProvider
   from .anthropic import AnthropicProvider
   
   PROVIDERS = {
       "openai": OpenAIProvider,
       "anthropic": AnthropicProvider,
   }
   
   def get_provider(name: str, **kwargs) -> LLMProvider:
       """Get LLM provider by name."""
       if name not in PROVIDERS:
           raise ValueError(f"Unknown provider: {name}")
       return PROVIDERS[name](**kwargs)
   ```

### Creating Custom Analysis Plugins

```python
# codelib/plugins/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..static import Codebase

class AnalysisPlugin(ABC):
    """Base class for analysis plugins."""
    
    @abstractmethod
    def analyze(self, codebase: Codebase) -> Dict[str, Any]:
        """Perform analysis on codebase."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

# Example plugin
class SecurityPlugin(AnalysisPlugin):
    """Security analysis plugin."""
    
    @property
    def name(self) -> str:
        return "security"
    
    def analyze(self, codebase: Codebase) -> Dict[str, Any]:
        # Perform security analysis
        vulnerabilities = self._find_vulnerabilities(codebase)
        return {
            "vulnerabilities": vulnerabilities,
            "risk_score": self._calculate_risk_score(vulnerabilities)
        }
```

## Performance Optimization

### Profiling

```python
# tools/profile.py
import cProfile
import pstats
from codelib import Codebase

def profile_codebase_loading():
    """Profile codebase loading performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run code to profile
    codebase = Codebase.from_path("/large/codebase")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

if __name__ == "__main__":
    profile_codebase_loading()
```

### Memory Optimization

```python
# Memory-efficient file processing
from typing import Iterator
import mmap

def process_large_file(file_path: str) -> Iterator[str]:
    """Process large files efficiently."""
    with open(file_path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b""):
                yield line.decode('utf-8')

# Lazy loading of codebase components
class LazyCodebase:
    """Codebase with lazy loading."""
    
    def __init__(self, path: str):
        self.path = path
        self._files = None
        self._symbols = None
    
    @property
    def files(self):
        if self._files is None:
            self._files = self._load_files()
        return self._files
    
    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = self._load_symbols()
        return self._symbols
```

### Caching Strategies

```python
# codelib/utils/cache.py
import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

class FileCache:
    """File-based caching for analysis results."""
    
    def __init__(self, cache_dir: str = "~/.cache/codelib"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Any:
        """Get cached value."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

def cached_analysis(cache_instance: FileCache):
    """Decorator for caching analysis results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance.cache_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
```

## Release Process

### Version Management

We use semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Checklist

1. **Update Version**
   ```bash
   # Update version in pyproject.toml
   version = "0.2.0"
   ```

2. **Update Changelog**
   ```markdown
   ## [0.2.0] - 2024-01-15
   
   ### Added
   - New semantic search capabilities
   - Support for additional LLM providers
   
   ### Changed
   - Improved performance for large codebases
   
   ### Fixed
   - Fixed memory leak in file processing
   ```

3. **Run Full Test Suite**
   ```bash
   pytest --cov=codelib
   pytest test/integration/
   ```

4. **Build and Test Package**
   ```bash
   python -m build
   python -m twine check dist/*
   ```

5. **Create Release**
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

6. **Publish to PyPI**
   ```bash
   python -m twine upload dist/*
   ```

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.12, 3.13]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,stats,builtin-impl]"
    
    - name: Run tests
      run: |
        pytest --cov=codelib --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Create GitHub issue for bugs/feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers directly for security issues

## Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

---

Thank you for contributing to CodeLib! Your efforts help make code analysis more accessible and powerful for everyone.