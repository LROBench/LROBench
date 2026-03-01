from src.operators.logical.base import LogicalOperator
from src.core.enums import OperandType, ImplType
import src.operators.physical as physical
import pandas as pd 

class LogicalMatch(LogicalOperator):
    def __init__(self, operand_type: OperandType, **kwargs):
        super().__init__(operand_type)
        self.kwargs = kwargs
        

    def execute(self, impl_type: ImplType, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs):
        if self.operand_type == OperandType.CELL:
            self.validate_inputs(['left_on', 'right_on'], kwargs)
        if self.operand_type == OperandType.COLUMN:
            self.validate_inputs(['example_num'], kwargs)

        thinking = kwargs['thinking'] if 'thinking' in kwargs else False

        if impl_type == ImplType.LLM_ALL:
            op = physical.LLMAllMatch(self.operand_type, thinking=thinking)
        elif impl_type == ImplType.LLM_ONE:
            op = physical.LLMOneMap(self.operand_type, thinking=thinking)
        elif impl_type == ImplType.LLM_SEMI: 
            op = physical.LLMSemiMap(self.operand_type, thinking=thinking)
        else:
            raise ValueError(f"Unsupported impl_type: {impl_type}")
        
        res = op.execute(condition = condition,
                    left_df = left_df.copy(), 
                    right_df = right_df.copy(), 
                    **kwargs)
        _total, _input, _output = op.get_consumed_tokens()
        self.total_tokens += _total
        self.input_tokens += _input
        self.output_tokens += _output
        return res
