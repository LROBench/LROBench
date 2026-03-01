from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.select_prompts as prompts
import pandas as pd
import json
from src.utils.parse import json_response_postprocess, list_wrapper
import asyncio

class LLMAllSelect(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ALL, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return self._row_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _row_execute(self, condition: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        depend_on = kwargs['depend_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")
        
        tmpdf = df[depend_on].reset_index()
        if not self.thinking:
            prompt = prompts.rows_one_call_prompt.format(tab = tmpdf.to_json(orient='records'), depend_on = depend_on, condition = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.rows_one_call_think_prompt.format(tab = tmpdf.to_json(orient='records'), depend_on = depend_on, condition = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        selected_idxes = pd.DataFrame(result)['index']
        return df.loc[selected_idxes]


    def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        example_num = kwargs['example_num']
        if example_num == 0:
            example_info = ""
        else:
            example_info = prompts.column_example_prompt.format(column_names = list_wrapper(df.columns), 
                                                                    column_values = df[:example_num].to_json(),
                                                                    indicator = "")
        if not self.thinking:
            prompt = prompts.columns_one_call_prompt.format(col_list=list_wrapper(df.columns),
                                                            condition=condition, 
                                                            extra_prompt=example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
            return df[result]
        else:
            prompt = prompts.columns_one_call_think_prompt.format(col_list=list_wrapper(df.columns),
                                                            condition=condition, 
                                                            extra_prompt=example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            cols =  result['result']
            cols =[ col.strip('{').strip('}') for col in cols]
            return df[cols]

    
    def _table_execute(self, condition: str, df: list, **kwargs) -> list:
        table_names = kwargs['table_names']
        example_num = kwargs['example_num']

        table_col_prompt = "- Input: \n"
        for (table, table_df) in zip(table_names, df):
            _cols = list_wrapper(table_df.columns)
            _prompt = f"Table '{table}' have columns '{_cols}'"
            table_col_prompt += _prompt + "\n"
        
        example_info = ""
        if example_num > 0:
            for (table, table_df) in zip(table_names, df):
                _prompt = prompts.column_example_prompt.format(column_names = list_wrapper(table_df.columns), 
                                                                    column_values = table_df[:example_num].to_json(),
                                                                    indicator = f"in Table '{table}'")
                example_info += _prompt + "\n"

        if not self.thinking:
            prompt = prompts.tables_one_call_prompt.format(tables=table_names, 
                                                condition=condition, 
                                                table_cols=table_col_prompt, 
                                                extra_prompt=example_info)
            
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
            return result
        else:
            prompt = prompts.tables_one_call_think_prompt.format(tables=table_names, 
                                                condition=condition, 
                                                table_cols=table_col_prompt, 
                                                extra_prompt=example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            return result['result']


class LLMOneSelect(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONE, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return asyncio.run(self._row_execute(condition, df, **kwargs))
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    async def _row_execute(self, condition: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        depend_on = kwargs['depend_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")
        
        semaphore = asyncio.Semaphore(10)
        async def process_row(row):
            async with semaphore:
                _row = json.dumps(row[depend_on].to_dict())
                if not self.thinking:
                    prompt = prompts.row_one_judge_prompt.format(
                        row=_row, condition=condition, extra_prompt=""
                    )
                    flag = await self.client.async_call([{'role': 'user', 'content': prompt}])
                else:
                    prompt = prompts.row_one_judge_think_prompt.format(
                        row=_row, condition=condition, extra_prompt=""
                    )
                    response = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    response = json_response_postprocess(response)[0]
                    # print(response)
                    flag = response['result']
                return row if int(flag) == 1 else None

        tasks = [process_row(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        filtered_rows = [row for row in results if row is not None]
        return pd.DataFrame(filtered_rows)
    

    def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        example_num = kwargs['example_num']
        columns = df.columns.values

        result = []
        for col in columns:
            if example_num == 0:
                example_info = ""
            else:
                example_info = prompts.column_example_prompt.format(column_names = col, 
                                                                    column_values = df[[col]][:example_num].to_json(),
                                                                    indicator = "")
            if not self.thinking:
                prompt = prompts.column_one_judge_prompt.format(row=col, condition=condition, extra_prompt=example_info)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
            else:
                prompt = prompts.column_one_judge_think_prompt.format(row=col, condition=condition, extra_prompt=example_info)
                response = self.client.call([{'role': 'user', 'content': prompt}])
                response = json_response_postprocess(response)[0]
                print(response)
                flag = response['result']
            if int(flag) == 1:
                result.append(col)
        return df[result]
            
    def _table_execute(self, condition: str, df: list, **kwargs) -> list:
        table_names = kwargs['table_names']
        example_num = kwargs['example_num']

        result = []
        for table, table_df in zip(table_names, df):
            example_info = ""
            if example_num > 0:
                example_info = prompts.column_example_prompt.format(column_names = list_wrapper(table_df.columns), 
                                                                    column_values = table_df[:example_num].to_json(),
                                                                    indicator = f"in Table '{table}'")
            if not self.thinking:
                prompt = prompts.table_one_judge_prompt.format(row = table, condition = condition, extra_prompt = example_info)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
            else:
                prompt = prompts.table_one_judge_think_prompt.format(row = table, condition = condition, extra_prompt = example_info)
                response = self.client.call([{'role': 'user', 'content': prompt}])
                response = json_response_postprocess(response)[0]
                print(response)
                flag = response['result']
            if int(flag) == 1:
                result.append(table)
        return result
