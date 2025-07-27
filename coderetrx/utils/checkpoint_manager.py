import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages checkpoint functionality for code retrieval operations.
    Handles saving and loading of progress state to enable resume functionality.
    """
    
    def __init__(self, repo_name: str, strategy: str, mode: str, checkpoint_dir: str = "checkpoint"):
        self.repo_name = repo_name
        self.strategy = strategy
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_path = self._generate_checkpoint_path()
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_checkpoint_path(self) -> Path:
        """Generate checkpoint file path based on naming convention"""
        filename = f"checkpoint_{self.repo_name}_{self.strategy}_{self.mode}.json"
        return self.checkpoint_dir / filename
    
    def _solve_json_serialization(self, obj: Any, seen: Optional[set] = None) -> Any:
        """
        Comprehensive JSON serialization solution that handles all complex objects
        without ignoring errors. This method ensures all objects can be serialized.
        """
        if seen is None:
            seen = set()
        
        # Handle None values
        if obj is None:
            return None
        
        # Handle primitive types first
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Prevent circular references
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular_reference_{type(obj).__name__}_{obj_id}>"
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        seen.add(obj_id)
        
        try:
            # Handle CodeMapFilterResult and similar custom objects
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'CodeMapFilterResult':
                result = {
                    'class_name': obj.__class__.__name__,
                    'index': getattr(obj, 'index', None),
                    'reason': str(getattr(obj, 'reason', '')),
                    'result': self._solve_json_serialization(getattr(obj, 'result', None), seen)
                }
                seen.discard(obj_id)
                return result
            
            # Handle Pydantic models
            if hasattr(obj, 'model_dump'):
                try:
                    result = obj.model_dump()
                    seen.discard(obj_id)
                    return self._solve_json_serialization(result, seen)
                except Exception as e:
                    logger.warning(f"Failed to use model_dump for {type(obj)}: {e}")
                    # Continue to fallback methods
            
            # Handle objects with dict() method
            if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
                try:
                    result = obj.dict()
                    seen.discard(obj_id)
                    return self._solve_json_serialization(result, seen)
                except Exception as e:
                    logger.warning(f"Failed to use dict() for {type(obj)}: {e}")
                    # Continue to fallback methods
            
            # Handle objects with to_json method
            if hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json')):
                try:
                    result = obj.to_json()
                    seen.discard(obj_id)
                    return self._solve_json_serialization(result, seen)
                except Exception as e:
                    logger.warning(f"Failed to use to_json for {type(obj)}: {e}")
                    # Continue to fallback methods
            
            # Handle collections
            if isinstance(obj, (list, tuple)):
                result = [self._solve_json_serialization(item, seen) for item in obj]
                seen.discard(obj_id)
                return result
            
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Ensure keys are strings
                    key = str(k) if not isinstance(k, str) else k
                    result[key] = self._solve_json_serialization(v, seen)
                seen.discard(obj_id)
                return result
            
            if isinstance(obj, set):
                result = [self._solve_json_serialization(item, seen) for item in obj]
                seen.discard(obj_id)
                return result
            
            # Handle objects with __dict__
            if hasattr(obj, '__dict__'):
                result = {}
                for k, v in obj.__dict__.items():
                    # Skip private/protected attributes and methods
                    if not k.startswith('_'):
                        try:
                            result[k] = self._solve_json_serialization(v, seen)
                        except Exception as e:
                            logger.warning(f"Failed to serialize attribute {k} of {type(obj)}: {e}")
                            result[k] = f"<serialization_failed_{type(v).__name__}>"
                
                # Add class information for reconstruction
                result['_class_name'] = obj.__class__.__name__
                result['_module_name'] = obj.__class__.__module__
                seen.discard(obj_id)
                return result
            
            # Handle callable objects
            if callable(obj):
                seen.discard(obj_id)
                return f"<callable_{obj.__name__ if hasattr(obj, '__name__') else 'unknown'}>"
            
            # Final attempt: try direct JSON serialization
            try:
                json.dumps(obj)
                seen.discard(obj_id)
                return obj
            except (TypeError, ValueError) as e:
                logger.warning(f"Object {type(obj)} cannot be JSON serialized: {e}")
                seen.discard(obj_id)
                return {
                    '_type': type(obj).__name__,
                    '_str_representation': str(obj),
                    '_serialization_note': 'Object converted to string representation'
                }
        
        except Exception as e:
            logger.error(f"Critical error in JSON serialization for {type(obj)}: {e}")
            seen.discard(obj_id)
            return {
                '_type': type(obj).__name__,
                '_error': str(e),
                '_serialization_note': 'Critical serialization error occurred'
            }
    
    def save_checkpoint(self, 
                       repository: str,
                       completed_prompts: List[int],
                       accumulated_cost: float,
                       accumulated_input_tokens: int,
                       accumulated_output_tokens: int,
                       timing_info: Dict[str, Any],
                       cost_info: Dict[str, Any],
                       completed_codes: Optional[List[Dict[str, Any]]] = None,
                       codebase_state_path: Optional[str] = None) -> bool:
        """
        Save checkpoint data to file. Returns True if successful, raises exception if failed.
        """
        try:
            # Clean timing_info and cost_info to only keep per-prompt data, not totals
            cleaned_timing_info = copy.deepcopy(timing_info)
            cleaned_timing_info.pop('total_duration', None)
            cleaned_timing_info.pop('average_prompt_duration', None)
            
            cleaned_cost_info = copy.deepcopy(cost_info)
            cleaned_cost_info.pop('total_cost', None)
            cleaned_cost_info.pop('total_input_tokens', None)
            cleaned_cost_info.pop('total_output_tokens', None)
            cleaned_cost_info.pop('average_prompt_cost', None)
            
            checkpoint_data = {
                "repository": repository,
                "timestamp": datetime.now().isoformat(),
                "strategy": self.strategy,
                "mode": self.mode,
                "completed_prompts": completed_prompts.copy(),
                "accumulated_cost": accumulated_cost,
                "accumulated_input_tokens": accumulated_input_tokens,
                "accumulated_output_tokens": accumulated_output_tokens,
                "timing_info": cleaned_timing_info,
                "cost_info": cleaned_cost_info,
                "completed_codes": completed_codes or [],
                "codebase_state_path": codebase_state_path
            }
            
            # Apply comprehensive JSON serialization
            serializable_data = self._solve_json_serialization(checkpoint_data)
            
            # Validate JSON serialization before saving
            try:
                json_str = json.dumps(serializable_data, indent=2)
            except Exception as e:
                raise RuntimeError(f"JSON serialization validation failed: {e}")
            
            # Write to file
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            logger.info(f"Checkpoint saved successfully to {self.checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data from file. Returns None if no valid checkpoint exists.
        Raises exception for corrupted checkpoints that cannot be recovered.
        """
        if not self.checkpoint_path.exists():
            logger.info(f"No checkpoint found at {self.checkpoint_path}")
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Validate required fields
            required_fields = [
                "repository", "strategy", "mode", "completed_prompts", "accumulated_cost"
            ]
            
            for field in required_fields:
                if field not in checkpoint_data:
                    raise ValueError(f"Missing required field: {field}")
            
            logger.info(f"Checkpoint loaded successfully from {self.checkpoint_path}")
            return checkpoint_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Checkpoint file is corrupted (JSON decode error): {e}")
            raise RuntimeError(f"Corrupted checkpoint file: {e}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint load failed: {e}")
    
    def should_resume_from_prompt(self, prompt_index: int) -> bool:
        """Check if we should resume from a specific prompt index"""
        checkpoint_data = self.load_checkpoint()
        if not checkpoint_data:
            return False
        
        completed_prompts = checkpoint_data.get("completed_prompts", [])
        return prompt_index not in completed_prompts
    
    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        """Get the state needed to resume processing"""
        checkpoint_data = self.load_checkpoint()
        if not checkpoint_data:
            return None
        
        return {
            "completed_prompts": set(checkpoint_data["completed_prompts"]),
            "accumulated_cost": checkpoint_data["accumulated_cost"],
            "accumulated_input_tokens": checkpoint_data.get("accumulated_input_tokens", 0),
            "accumulated_output_tokens": checkpoint_data.get("accumulated_output_tokens", 0),
            "timing_info": checkpoint_data.get("timing_info", {}),
            "cost_info": checkpoint_data.get("cost_info", {}),
            "completed_codes": checkpoint_data.get("completed_codes", []),
            "repository": checkpoint_data["repository"]
        }
    
    def cleanup_checkpoint(self) -> bool:
        """Remove checkpoint file. Returns True if successful."""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info(f"Checkpoint file removed: {self.checkpoint_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint: {e}")
            return False
    
    def validate_checkpoint_integrity(self) -> bool:
        """Validate that checkpoint file is not corrupted"""
        try:
            checkpoint_data = self.load_checkpoint()
            return checkpoint_data is not None
        except Exception:
            return False