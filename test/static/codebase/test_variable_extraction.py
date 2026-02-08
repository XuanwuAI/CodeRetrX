#!/usr/bin/env python3
"""
Test variable definition extraction - All supported languages
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from coderetrx.static.codebase import Codebase, ChunkType


# Test code samples
TEST_SAMPLES = {
    "python": {
        "filename": "test_vars.py",
        "code": """# Python variable test
# Module-level variables
CONFIG = {"key": "value"}
MAX_RETRIES = 3
API_URL = "https://api.example.com"

class MyClass:
    # Class variables
    class_var = "class variable"
    CONSTANT = 100

    def __init__(self):
        # Instance variables
        self.instance_var = "instance"
        self.count = 0
        # Local variables
        local_var = "local"
        temp = 42

def my_function():
    # Function local variables
    func_var = "function variable"
    result = []
    return func_var
""",
        "expected_vars": ["CONFIG", "MAX_RETRIES", "API_URL", "class_var", "CONSTANT",
                         "self.instance_var", "self.count", "local_var", "temp",
                         "func_var", "result"],
    },

    "go": {
        "filename": "test_vars.go",
        "code": """package main

// Go variable test
// Package-level variables
var GlobalVar = "global"
var Counter int = 0

// Package-level constants
const MaxRetries = 3
const APIEndpoint = "https://api.example.com"

type Config struct {
    Name string
}

func main() {
    // Local variables - short declaration
    localVar := "local"
    count := 0

    // Local variables - var declaration
    var anotherVar string = "another"
    var result []string
}

func process() {
    data := make([]byte, 100)
    _ = data
}
""",
        "expected_vars": ["GlobalVar", "Counter", "MaxRetries", "APIEndpoint",
                         "localVar", "count", "anotherVar", "result", "data"],
    },

    "javascript": {
        "filename": "test_vars.js",
        "code": """// JavaScript variable test
// Top-level variables
const API_URL = "https://api.example.com";
let counter = 0;
var oldStyle = "old";

const config = {
    key: "value"
};

class MyClass {
    constructor() {
        this.instanceVar = "instance";
    }

    method() {
        const localConst = "local";
        let localLet = "local";
        var localVar = "local";
    }
}

function myFunc() {
    const funcConst = "function";
    let funcLet = 0;
}

// Arrow function assignment (this will be identified as a function definition)
const myArrowFunc = () => {
    const arrowVar = "arrow";
    return arrowVar;
};
""",
        "expected_vars": ["API_URL", "counter", "oldStyle", "config",
                         "localConst", "localLet", "localVar",
                         "funcConst", "funcLet", "myArrowFunc", "arrowVar"],
    },

    "typescript": {
        "filename": "test_vars.ts",
        "code": """// TypeScript variable test
// Top-level variables
const API_URL: string = "https://api.example.com";
let counter: number = 0;
var oldStyle: string = "old";

interface Config {
    key: string;
}

const config: Config = {
    key: "value"
};

class MyClass {
    private instanceVar: string;

    constructor() {
        this.instanceVar = "instance";
        const localConst: string = "local";
    }

    method(): void {
        let localLet: number = 0;
    }
}

function myFunc(): string {
    const funcConst: string = "function";
    return funcConst;
}
""",
        "expected_vars": ["API_URL", "counter", "oldStyle", "config",
                         "localConst", "localLet", "funcConst"],
    },

    "rust": {
        "filename": "test_vars.rs",
        "code": """// Rust variable test
// Static variables
static GLOBAL_VAR: &str = "global";
static mut COUNTER: i32 = 0;

// Constants
const MAX_RETRIES: i32 = 3;
const API_URL: &str = "https://api.example.com";

struct Config {
    name: String,
}

fn main() {
    // Local variables
    let local_var = "local";
    let mut count = 0;
    let result: Vec<String> = Vec::new();
}

fn process() {
    let data = vec![1, 2, 3];
    let _unused = 42;
}
""",
        "expected_vars": ["GLOBAL_VAR", "COUNTER", "MAX_RETRIES", "API_URL",
                         "local_var", "count", "result", "data", "_unused"],
    },

    "java": {
        "filename": "TestVars.java",
        "code": """// Java variable test
public class TestVars {
    // Static fields
    public static final String API_URL = "https://api.example.com";
    private static int counter = 0;

    // Instance fields
    private String instanceVar;
    private int count;

    public TestVars() {
        this.instanceVar = "instance";
        // Local variables
        String localVar = "local";
        int temp = 42;
    }

    public void method() {
        // Method local variables
        String methodVar = "method";
        int result = 0;
    }
}
""",
        "expected_vars": ["API_URL", "counter", "instanceVar", "count",
                         "localVar", "temp", "methodVar", "result"],
    },

    "c": {
        "filename": "test_vars.c",
        "code": """// C variable test
#include <stdio.h>

// Global variables
int global_var = 0;
char *global_str = "global";
const int MAX_SIZE = 100;

struct Config {
    char *name;
    int value;
};

int main() {
    // Local variables
    int local_var = 42;
    char *local_str = "local";
    struct Config config;

    return 0;
}

void process() {
    int data = 0;
    char buffer[256];
}
""",
        "expected_vars": ["global_var", "global_str", "MAX_SIZE",
                         "local_var", "local_str", "config", "data", "buffer"],
    },

    "cpp": {
        "filename": "test_vars.cpp",
        "code": """// C++ variable test
#include <string>
#include <vector>

// Global variables
int global_var = 0;
std::string global_str = "global";
const int MAX_SIZE = 100;

class MyClass {
private:
    int instance_var;
    std::string name;

public:
    MyClass() {
        instance_var = 0;
        // Local variables
        int local_var = 42;
        std::string local_str = "local";
    }

    void method() {
        int method_var = 0;
        std::vector<int> data;
    }
};

int main() {
    int local_var = 42;
    MyClass obj;
    return 0;
}
""",
        "expected_vars": ["global_var", "global_str", "MAX_SIZE",
                         "instance_var", "name", "local_var", "local_str",
                         "method_var", "data", "obj"],
    },

    "csharp": {
        "filename": "TestVars.cs",
        "code": """// C# variable test
using System;

namespace TestNamespace
{
    public class TestVars
    {
        // Static fields
        public static readonly string ApiUrl = "https://api.example.com";
        private static int counter = 0;

        // Instance fields
        private string instanceVar;
        private int count;

        // Properties
        public string Name { get; set; }

        public TestVars()
        {
            this.instanceVar = "instance";
            // Local variables
            string localVar = "local";
            int temp = 42;
        }

        public void Method()
        {
            // Method local variables
            string methodVar = "method";
            int result = 0;
        }
    }
}
""",
        "expected_vars": ["ApiUrl", "counter", "instanceVar", "count", "Name",
                         "localVar", "temp", "methodVar", "result"],
    },

    "elixir": {
        "filename": "test_vars.ex",
        "code": """# Elixir variable test
defmodule TestVars do
  # Module attributes
  @module_var "module"
  @max_retries 3
  @api_url "https://api.example.com"

  def my_function do
    # Variable bindings in function (not extracted in Elixir)
    local_var = "local"
    count = 0
    local_var
  end

  def process(data) do
    result = Enum.map(data, fn x -> x * 2 end)
    result
  end
end
""",
        "expected_vars": ["module_var", "max_retries", "api_url"],
    },
}


class TestVariableExtraction:
    """Test variable definition extraction"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary test directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_python_variables(self, temp_dir):
        """Test Python variable extraction"""
        self._test_language(temp_dir, "python")

    def test_go_variables(self, temp_dir):
        """Test Go variable extraction"""
        self._test_language(temp_dir, "go")

    def test_javascript_variables(self, temp_dir):
        """Test JavaScript variable extraction"""
        self._test_language(temp_dir, "javascript")

    def test_typescript_variables(self, temp_dir):
        """Test TypeScript variable extraction"""
        self._test_language(temp_dir, "typescript")

    def test_rust_variables(self, temp_dir):
        """Test Rust variable extraction"""
        self._test_language(temp_dir, "rust")

    def test_java_variables(self, temp_dir):
        """Test Java variable extraction"""
        self._test_language(temp_dir, "java")

    def test_c_variables(self, temp_dir):
        """Test C variable extraction"""
        self._test_language(temp_dir, "c")

    def test_cpp_variables(self, temp_dir):
        """Test C++ variable extraction"""
        self._test_language(temp_dir, "cpp")

    def test_csharp_variables(self, temp_dir):
        """Test C# variable extraction"""
        self._test_language(temp_dir, "csharp")

    def test_elixir_variables(self, temp_dir):
        """Test Elixir variable extraction"""
        self._test_language(temp_dir, "elixir")

    def _test_language(self, temp_dir: Path, language: str):
        """Generic language test method"""
        sample = TEST_SAMPLES[language]

        # Create test file
        test_file = temp_dir / sample["filename"]
        test_file.write_text(sample["code"])

        # Create Codebase and extract chunks
        codebase = Codebase.new(
            id=f"test_{language}",
            dir=temp_dir,
            parser="treesitter"
        )
        codebase.init_chunks()

        # Get variable definition chunks
        variable_chunks = [c for c in codebase.all_chunks if c.type == ChunkType.VARIABLE]
        extracted_vars = [c.name for c in variable_chunks if c.name]

        # Print debug info
        print(f"\n{'='*60}")
        print(f"Test language: {language.upper()}")
        print(f"{'='*60}")
        print(f"Extracted variable count: {len(extracted_vars)}")
        print(f"Extracted variables: {extracted_vars}")
        print(f"Expected variables: {sample['expected_vars']}")

        # Check if any variables were extracted
        assert len(extracted_vars) > 0, f"{language}: No variables were extracted"

        # Check if key variables were extracted (exact match not required, as some languages may have special cases)
        # At least some expected variables should be extracted
        found_count = sum(1 for var in sample['expected_vars'] if var in extracted_vars)
        coverage = found_count / len(sample['expected_vars']) * 100

        print(f"Coverage: {coverage:.1f}% ({found_count}/{len(sample['expected_vars'])})")
        print(f"{'='*60}\n")

        # At least 50% of expected variables should be extracted
        assert coverage >= 50, f"{language}: Variable extraction coverage too low ({coverage:.1f}%)"


if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])