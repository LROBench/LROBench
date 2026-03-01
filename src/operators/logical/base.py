from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Optional
from dataclasses import dataclass
from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator


class LogicalOperator(ABC):
    def __init__(self, operand_type: OperandType):
        self.operand_type = operand_type
        self.children: List[LogicalOperator] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
    
    @abstractmethod
    def execute(self, impl_type: ImplType):
        pass
    
    def add_child(self, child: 'LogicalOperator'):
        self.children.append(child)

    def validate_inputs(self, required_args: list, kwargs: Dict[str, Any]):
        missing_args = [arg for arg in required_args if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}")
        
    def get_tokens(self):
        return self.total_tokens, self.input_tokens, self.output_tokens
    
