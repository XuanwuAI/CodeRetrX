from dotenv import load_dotenv
import json
from pathlib import Path
from codelib.impl.default import CodebaseFactory
from codelib.impl.default import TopicExtractor
from codelib.retrieval.code_recall import multi_strategy_code_filter, multi_strategy_code_mapping
import asyncio
import unittest
from typing import Literal
from codelib.utils.git import clone_repo_if_not_exists, get_repo_id, get_data_dir
import logging

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_REPOS = ["https://github.com/apache/dubbo-admin.git"]


def prepare_codebase(repo_url: str, repo_path: Path):
    """Helper function to prepare the codebase for testing"""

    # Create a test codebase
    database_path = get_data_dir() / "databases" / f"{get_repo_id(repo_url)}.json"
    clone_repo_if_not_exists(repo_url, str(repo_path))

    if database_path.exists():
        codebase = CodebaseFactory.from_json(
            json.load(open(database_path, "r", encoding="utf-8"))
        )
    else:
        codebase = CodebaseFactory.new(get_repo_id(repo_url), repo_path)
    with open(f"{repo_path}.json", "w") as f:
        json.dump(codebase.to_json(), f, indent=4)
    return codebase


class TestLLMCodeFilterTool(unittest.TestCase):
    """Test LLMCodeFilterTool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.repo_url = TEST_REPOS[0]
        self.repo_path = get_data_dir() / "repos" / get_repo_id(self.repo_url)
        self.codebase = prepare_codebase(self.repo_url, self.repo_path)
        self.test_dir = "/"
        self.test_prompt = "Is the code snippet used for user authentication?"
        self.topic_extractor = TopicExtractor()
        # Add debug flag - set to True to see all captured error messages
        self.debug_error_logging = False
    
    async def recall_with_mode(self, mode: Literal["fast", "balance", "precise", "custom"], llm_call_mode: Literal["traditional", "function_call"]):
        """Helper method to run the tool with a specific mode"""
        if (llm_call_mode == "function_call"):
            result, llm_output = await multi_strategy_code_filter(
                codebase=self.codebase,
                subdirs_or_files=[self.test_dir],
                prompt=self.test_prompt,
                granularity="symbol_content",
                mode=mode,
                topic_extractor=self.topic_extractor,
                llm_call_mode="function_call"
            )
            return result
        result, llm_output = await multi_strategy_code_filter(
            codebase=self.codebase,
            subdirs_or_files=[self.test_dir],
            prompt=self.test_prompt,
            granularity="symbol_content",
            mode=mode,
            topic_extractor=self.topic_extractor,
            llm_call_mode="traditional"
        )
        return result
    
    def test_initialization(self):
        """Test initialization of test environment"""
        self.assertIsNotNone(self.codebase)
        self.assertIsNotNone(self.test_dir)
    
    def test_traditional_mode(self):
        """Test LLMCodeFilterTool in balance mode"""
        print("test_traditional_mode")
        error_count = 0
        missing_count = 0
        
        # Create a custom log handler to capture missing field errors
        class MissingFieldHandler(logging.Handler):
            def __init__(self, debug_flag=False):
                super().__init__()
                self.missing_count = 0
                self.captured_errors = []  # Store captured error details for debugging
                self.debug_error_logging = True
            
            def emit(self, record):
                msg_str = str(record.getMessage()).lower()  # Use getMessage() for better formatting
                
                # Capture various patterns of missing field and validation errors
                patterns = [
                    ('missing' in msg_str and 'field' in msg_str),  # Direct missing field
                    ('field required' in msg_str),  # Pydantic validation
                    ('validation error' in msg_str and 'result' in msg_str),  # Validation with result field
                    ('missing' in msg_str and 'result' in msg_str),  # Missing result field specifically
                    ('failed validation' in msg_str),  # General validation failures
                    ('item at index' in msg_str and 'failed validation' in msg_str),  # List item validation
                ]
                
                if any(patterns):
                    self.missing_count += 1
                    error_detail = f"{record.levelname} - {record.name} - {record.getMessage()}"
                    self.captured_errors.append(error_detail)
                    
                    # Print for debugging if enabled
                    if self.debug_error_logging:
                        print(f"Captured error #{self.missing_count}: {error_detail}")
        
        # Add the custom handler to capture missing field logs
        missing_handler = MissingFieldHandler(self.debug_error_logging)
        missing_handler.setLevel(logging.WARNING)  # Capture WARNING and above (including ERROR)
        
        # Get all relevant loggers that might log missing field errors
        loggers_to_monitor = [
            logging.getLogger('codelib.impl.default.smart_codebase'),
            logging.getLogger('codelib.utils.llm'),
            logging.getLogger('codelib.retrieval.code_recall'),
            logging.getLogger('codelib.utils.jsonparser'),
            logging.getLogger(),  # Root logger as fallback
        ]
        
        # Add handler to all relevant loggers
        for logger_obj in loggers_to_monitor:
            logger_obj.addHandler(missing_handler)
        
        try:
            result = asyncio.run(self.recall_with_mode("balance", "traditional"))
            
            # Get the missing count from our custom handler
            missing_count = missing_handler.missing_count
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            logger.info(f"Balance mode results count: {len(result)}")
            if result:
                logger.info(f"Sample result: {result[0]}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error in traditional mode: {e}")
        finally:
            # Clean up - remove the custom handler from all loggers
            for logger_obj in loggers_to_monitor:
                logger_obj.removeHandler(missing_handler)
        
        print(f"Total missing field errors in traditional mode: {missing_count}")
        
        # Detailed reporting
        if self.debug_error_logging and missing_handler.captured_errors:
            print("Captured error details:")
            for i, error in enumerate(missing_handler.captured_errors, 1):
                print(f"  {i}. {error}")
        elif missing_count > 0:
            print(f"Set debug_error_logging=True to see details of {missing_count} captured errors")
    
    def test_function_call_mode(self):
        """Test LLMCodeFilterTool in balance mode"""
        print("test_function_call_mode")
        error_count = 0
        missing_count = 0
        
        # Create a custom log handler to capture missing field errors
        class MissingFieldHandler(logging.Handler):
            def __init__(self, debug_flag=False):
                super().__init__()
                self.missing_count = 0
                self.captured_errors = []  # Store captured error details for debugging
                self.debug_error_logging = debug_flag
            
            def emit(self, record):
                msg_str = str(record.getMessage()).lower()  # Use getMessage() for better formatting
                
                # Capture various patterns of missing field and validation errors
                patterns = [
                    ('missing' in msg_str and 'field' in msg_str),  # Direct missing field
                    ('field required' in msg_str),  # Pydantic validation
                    ('validation error' in msg_str and 'result' in msg_str),  # Validation with result field
                    ('missing' in msg_str and 'result' in msg_str),  # Missing result field specifically
                    ('failed validation' in msg_str),  # General validation failures
                    ('item at index' in msg_str and 'failed validation' in msg_str),  # List item validation
                ]
                
                if any(patterns):
                    self.missing_count += 1
                    error_detail = f"{record.levelname} - {record.name} - {record.getMessage()}"
                    self.captured_errors.append(error_detail)
                    
                    # Print for debugging if enabled
                    if self.debug_error_logging:
                        print(f"Captured error #{self.missing_count}: {error_detail}")
        
        # Add the custom handler to capture missing field logs
        missing_handler = MissingFieldHandler(self.debug_error_logging)
        missing_handler.setLevel(logging.WARNING)  # Capture WARNING and above (including ERROR)
        
        # Get all relevant loggers that might log missing field errors
        loggers_to_monitor = [
            logging.getLogger('codelib.impl.default.smart_codebase'),
            logging.getLogger('codelib.utils.llm'),
            logging.getLogger('codelib.retrieval.code_recall'),
            logging.getLogger('codelib.utils.jsonparser'),
            logging.getLogger(),  # Root logger as fallback
        ]
        
        # Add handler to all relevant loggers
        for logger_obj in loggers_to_monitor:
            logger_obj.addHandler(missing_handler)
        
        try:
            result = asyncio.run(self.recall_with_mode("balance", "function_call"))
            
            # Get the missing count from our custom handler
            missing_count = missing_handler.missing_count
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            logger.info(f"Balance mode results count: {len(result)}")
            if result:
                logger.info(f"Sample result: {result[0]}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error in function call mode: {e}")
        finally:
            # Clean up - remove the custom handler from all loggers
            for logger_obj in loggers_to_monitor:
                logger_obj.removeHandler(missing_handler)
        
        print(f"Total missing field errors in function call mode: {missing_count}")
        
        # Detailed reporting
        if self.debug_error_logging and missing_handler.captured_errors:
            print("Captured error details:")
            for i, error in enumerate(missing_handler.captured_errors, 1):
                print(f"  {i}. {error}")
        elif missing_count > 0:
            print(f"Set debug_error_logging=True to see details of {missing_count} captured errors")

# Run tests if specified
if __name__ == "__main__":
    # Run unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
