from langchain.output_parsers.json import SimpleJsonOutputParser, parse_json_markdown
from pydantic import BaseModel
from langchain_core.outputs import Generation
from typing import Iterable, List, Any, Type, Optional
import json_repair  # For repairing malformed JSON strings
import functools  # For partial function application
import logging  # For error logging and debugging

# Initialize logger for tracking parsing errors and debug information
logger = logging.getLogger(__name__)


class TolerantJsonParser(SimpleJsonOutputParser):
    """
    A robust JSON parser with multiple fallback strategies for handling malformed or partial JSON input.

    Features:
    - Multiple JSON parsing attempts with different repair strategies
    - Pydantic model validation integration
    - Error recovery and detailed logging
    - Flexible preprocessing and postprocessing hooks

    Attributes:
        starting_chars (set): Valid starting characters for JSON structures ({, [)
        return_basemodel (bool): Flag to enable Pydantic model validation
        pydantic_object (Type[BaseModel]): Pydantic model class for validation
    """

    # Characters that indicate potential JSON structures
    starting_chars: set[str] = {"[", "{"}
    # Configuration flags for Pydantic model handling
    return_basemodel: bool = False
    pydantic_object: Optional[Type[BaseModel]] = None

    def preprocess(self, text: str) -> str:
        """Basic text normalization before parsing attempts.

        Args:
            text (str): Raw input text potentially containing JSON

        Returns:
            str: Trimmed text with leading/trailing whitespace removed
        """
        return text.strip()

    def postprocess(self, obj: Any) -> Any:
        """Post-processing hook for modifying parsed JSON objects.

        Can be overridden in subclasses for custom transformations.

        Args:
            obj (Any): Successfully parsed JSON object

        Returns:
            Any: Processed JSON object
        """
        return obj

    def potential_json_substrings(self, text: str) -> Iterable[str]:
        """Generate candidate JSON substrings from input text.

        Yields:
            str: Potential JSON fragments starting with valid JSON characters

        Example:
            For input "Text before {\"key\": \"value\"}", yields:
            - Full text
            - {\"key\": \"value\"}
        """
        yield text  # First try the full text
        # Then try all substrings starting with JSON characters
        for i in range(len(text)):
            if text[i] in self.starting_chars:
                yield text[i:]

    def is_valid_json(self, obj: Any) -> bool:
        """Validation hook for parsed JSON objects.

        Can be overridden in subclasses for custom validation logic.

        Args:
            obj (Any): Parsed JSON object to validate

        Returns:
            bool: True if object is considered valid, False otherwise
        """
        return True  # Default implementation accepts any parsed JSON

    def is_valid_model(self, obj: BaseModel) -> bool:
        """Validation hook for Pydantic models.

        Can be overridden in subclasses for custom model validation.

        Args:
            obj (BaseModel): Parsed Pydantic model to validate

        Returns:
            bool: True if model is valid, False otherwise
        """
        return True  # Default implementation accepts any valid Pydantic model

    @staticmethod
    def markdown_json_repair(text: str):
        """Specialized parser for JSON embedded in markdown code blocks.

        Combines markdown parsing with JSON repair capabilities.

        Args:
            text (str): Input text potentially containing markdown-wrapped JSON

        Returns:
            Any: Parsed JSON object
        """
        return parse_json_markdown(
            text, parser=functools.partial(json_repair.loads, skip_json_loads=True)
        )

    def load_json(self, text: str):
        """Multi-strategy JSON loading pipeline.

        Attempts parsing with different strategies in sequence:
        1. Standard markdown JSON parsing
        2. Direct JSON repair
        3. Combined markdown/JSON repair

        Args:
            text (str): Input text to parse

        Yields:
            Any: Successfully parsed JSON objects from different strategies
        """
        parsers = [
            parse_json_markdown,  # Try standard markdown parsing first
            json_repair.loads,  # Fallback to direct JSON repair
            self.markdown_json_repair,  # Combined approach
        ]
        for parser in parsers:
            try:
                yield parser(text)
            except (ValueError, KeyError):
                pass  # Silently fail and try next parser

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """Main parsing workflow with error handling and validation.

        Args:
            result (List[Generation]): Input data containing text to parse
            partial (bool): Flag for partial parsing (unused in current implementation)

        Returns:
            Any: Parsed JSON object, validated Pydantic model, or None on failure

        Raises:
            Logs warnings and debug information on parsing failures
        """
        error_msgs = []
        prep_text = self.preprocess(result[0].text)

        # Try all potential JSON substrings
        for text in self.potential_json_substrings(prep_text):
            # Attempt different parsing strategies
            for json_obj in self.load_json(text):
                processed_obj = self.postprocess(json_obj)

                if not self.is_valid_json(processed_obj):
                    error_msgs.append(f"Validation failed for: {processed_obj}")
                    continue

                # Handle Pydantic model validation if configured
                if self.return_basemodel and self.pydantic_object:
                    try:
                        model_obj = self.pydantic_object.parse_obj(processed_obj)
                        if self.is_valid_model(model_obj):
                            return model_obj
                        error_msgs.append(f"Invalid model: {model_obj.dict()}")
                    except Exception as err:
                        error_msgs.append(
                            f"Model parsing error: {err} | Data: {processed_obj}"
                        )
                else:
                    return processed_obj  # Return raw JSON if model validation not required

        # Log final failure after all attempts
        logger.warning(
            f"JSON parsing failed after {len(error_msgs)} attempts. Input: {prep_text}"
        )
        logger.debug("Failure reasons:\n" + "\n".join(error_msgs))
        return None

    def parse(self, text: str) -> Any:
        """Convenience method for parsing raw text input.

        Args:
            text (str): Raw input text containing potential JSON

        Returns:
            Any: Parsed result or None
        """
        return self.parse_result([Generation(text=text)])
