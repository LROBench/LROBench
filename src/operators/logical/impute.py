from src.operators.logical.base import LogicalOperator
from src.core.enums import OperandType, ImplType
import src.operators.physical as physical
import pandas as pd 

class LogicalImpute(LogicalOperator):
    def __init__(self, operand_type: OperandType, **kwargs):
        super().__init__(operand_type)
        self.kwargs = kwargs
        
    def execute(self, impl_type: ImplType, condition: str, df: pd.DataFrame = None, **kwargs):
        if self.operand_type != OperandType.TABLE and df is None:
            raise ValueError(f"The input table is None.")
        
        thinking = kwargs['thinking'] if 'thinking' in kwargs else False
        
        # Directly execute for TABLE
        if self.operand_type == OperandType.TABLE:
            self.validate_inputs(['col_names'], kwargs)
            op = physical.LLMAllImpute(self.operand_type)
            return op.execute(condition = condition, 
                              df = df, 
                              **kwargs)

        # Validate params of CELL, COLUMN, ROW
        if self.operand_type == OperandType.CELL and impl_type == ImplType.LLM_ONE:
            self.validate_inputs(['example_num'], kwargs) 
        if self.operand_type == OperandType.COLUMN:
            self.validate_inputs(['depend_on', 'new_col'], kwargs)
        if self.operand_type == OperandType.ROW:
            self.validate_inputs(['example_num'], kwargs) 

        # Route to different physical operators
        if impl_type == ImplType.LLM_ALL:
            op = physical.LLMAllImpute(self.operand_type, thinking=thinking)
        elif impl_type == ImplType.LLM_ONE:
            op = physical.LLMOneImpute(self.operand_type, thinking=thinking)
        else:
            raise ValueError(f"Unsupported impl_type: {impl_type}")
        
        res =  op.execute(condition = condition, 
                          df = df.copy(), 
                          **kwargs)
        _total, _input, _output = op.get_consumed_tokens()
        self.total_tokens += _total
        self.input_tokens += _input
        self.output_tokens += _output
        return res