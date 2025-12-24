#!/usr/bin/env python3
"""
测试变量定义提取功能 - 所有支持的语言
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from coderetrx.static.codebase import Codebase, ChunkType


# 测试代码样本
TEST_SAMPLES = {
    "python": {
        "filename": "test_vars.py",
        "code": """# Python 变量测试
# 模块级变量
CONFIG = {"key": "value"}
MAX_RETRIES = 3
API_URL = "https://api.example.com"

class MyClass:
    # 类变量
    class_var = "class variable"
    CONSTANT = 100
    
    def __init__(self):
        # 实例变量
        self.instance_var = "instance"
        self.count = 0
        # 局部变量
        local_var = "local"
        temp = 42

def my_function():
    # 函数局部变量
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

// Go 变量测试
// 包级变量
var GlobalVar = "global"
var Counter int = 0

// 包级常量
const MaxRetries = 3
const APIEndpoint = "https://api.example.com"

type Config struct {
    Name string
}

func main() {
    // 局部变量 - 短声明
    localVar := "local"
    count := 0
    
    // 局部变量 - var 声明
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
        "code": """// JavaScript 变量测试
// 顶层变量
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

// 箭头函数赋值（这个会被识别为函数定义）
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
        "code": """// TypeScript 变量测试
// 顶层变量
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
        "code": """// Rust 变量测试
// 静态变量
static GLOBAL_VAR: &str = "global";
static mut COUNTER: i32 = 0;

// 常量
const MAX_RETRIES: i32 = 3;
const API_URL: &str = "https://api.example.com";

struct Config {
    name: String,
}

fn main() {
    // 局部变量
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
        "code": """// Java 变量测试
public class TestVars {
    // 静态字段
    public static final String API_URL = "https://api.example.com";
    private static int counter = 0;
    
    // 实例字段
    private String instanceVar;
    private int count;
    
    public TestVars() {
        this.instanceVar = "instance";
        // 局部变量
        String localVar = "local";
        int temp = 42;
    }
    
    public void method() {
        // 方法局部变量
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
        "code": """// C 变量测试
#include <stdio.h>

// 全局变量
int global_var = 0;
char *global_str = "global";
const int MAX_SIZE = 100;

struct Config {
    char *name;
    int value;
};

int main() {
    // 局部变量
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
        "code": """// C++ 变量测试
#include <string>
#include <vector>

// 全局变量
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
        // 局部变量
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
        "code": """// C# 变量测试
using System;

namespace TestNamespace
{
    public class TestVars
    {
        // 静态字段
        public static readonly string ApiUrl = "https://api.example.com";
        private static int counter = 0;
        
        // 实例字段
        private string instanceVar;
        private int count;
        
        // 属性
        public string Name { get; set; }
        
        public TestVars()
        {
            this.instanceVar = "instance";
            // 局部变量
            string localVar = "local";
            int temp = 42;
        }
        
        public void Method()
        {
            // 方法局部变量
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
        "code": """# Elixir 变量测试
defmodule TestVars do
  # 模块属性
  @module_var "module"
  @max_retries 3
  @api_url "https://api.example.com"
  
  def my_function do
    # 函数内变量绑定（Elixir 中不提取这些）
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
    """测试变量定义提取功能"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时测试目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_python_variables(self, temp_dir):
        """测试 Python 变量提取"""
        self._test_language(temp_dir, "python")
    
    def test_go_variables(self, temp_dir):
        """测试 Go 变量提取"""
        self._test_language(temp_dir, "go")
    
    def test_javascript_variables(self, temp_dir):
        """测试 JavaScript 变量提取"""
        self._test_language(temp_dir, "javascript")
    
    def test_typescript_variables(self, temp_dir):
        """测试 TypeScript 变量提取"""
        self._test_language(temp_dir, "typescript")
    
    def test_rust_variables(self, temp_dir):
        """测试 Rust 变量提取"""
        self._test_language(temp_dir, "rust")
    
    def test_java_variables(self, temp_dir):
        """测试 Java 变量提取"""
        self._test_language(temp_dir, "java")
    
    def test_c_variables(self, temp_dir):
        """测试 C 变量提取"""
        self._test_language(temp_dir, "c")
    
    def test_cpp_variables(self, temp_dir):
        """测试 C++ 变量提取"""
        self._test_language(temp_dir, "cpp")
    
    def test_csharp_variables(self, temp_dir):
        """测试 C# 变量提取"""
        self._test_language(temp_dir, "csharp")
    
    def test_elixir_variables(self, temp_dir):
        """测试 Elixir 变量提取"""
        self._test_language(temp_dir, "elixir")
    
    def _test_language(self, temp_dir: Path, language: str):
        """通用语言测试方法"""
        sample = TEST_SAMPLES[language]
        
        # 创建测试文件
        test_file = temp_dir / sample["filename"]
        test_file.write_text(sample["code"])
        
        # 创建 Codebase 并提取 chunks
        codebase = Codebase.new(
            id=f"test_{language}",
            dir=temp_dir,
            parser="treesitter"
        )
        codebase.init_chunks()
        
        # 获取变量定义 chunks
        variable_chunks = [c for c in codebase.all_chunks if c.type == ChunkType.VARIABLE]
        extracted_vars = [c.name for c in variable_chunks if c.name]
        
        # 打印调试信息
        print(f"\n{'='*60}")
        print(f"测试语言: {language.upper()}")
        print(f"{'='*60}")
        print(f"提取的变量数量: {len(extracted_vars)}")
        print(f"提取的变量: {extracted_vars}")
        print(f"期望的变量: {sample['expected_vars']}")
        
        # 检查是否提取到了变量
        assert len(extracted_vars) > 0, f"{language}: 没有提取到任何变量"
        
        # 检查关键变量是否被提取（不要求完全匹配，因为某些语言可能有特殊情况）
        # 至少应该提取到一些期望的变量
        found_count = sum(1 for var in sample['expected_vars'] if var in extracted_vars)
        coverage = found_count / len(sample['expected_vars']) * 100
        
        print(f"覆盖率: {coverage:.1f}% ({found_count}/{len(sample['expected_vars'])})")
        print(f"{'='*60}\n")
        
        # 至少应该提取到 50% 的期望变量
        assert coverage >= 50, f"{language}: 变量提取覆盖率过低 ({coverage:.1f}%)"


if __name__ == "__main__":
    # 直接运行测试
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
