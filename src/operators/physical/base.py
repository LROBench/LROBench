from abc import ABC, abstractmethod
from src.core.enums import ImplType, OperandType
import pandas as pd 
from src.utils.LLMCaller import LLMCaller


class PhysicalOperator(ABC):
    def __init__(self, implementation_type: ImplType, operand_type: OperandType, thinking: bool = False):
        self.implementation_type = implementation_type
        self.operand_type = operand_type
        self.client = LLMCaller()
        self.thinking = thinking

    @abstractmethod
    def execute(self, **kwargs) -> pd.DataFrame:
        pass
    
    def get_consumed_tokens(self):
        return self.client.get_total_tokens_used()
