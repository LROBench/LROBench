from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.impute_prompts as prompts
import pandas as pd
import json
from src.utils.parse import json_response_postprocess
import asyncio

class LLMAllImpute(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ALL, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.CELL:
            return self._cell_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.ROW:
            return self._row_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _cell_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        if not self.thinking:
            prompt = prompts.cells_one_call_prompt.format(tab = df.to_json(orient='records'), condition = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.cells_one_call_think_prompt.format(tab = df.to_json(orient='records'), condition = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']           
        result = pd.DataFrame(result)
        result.index = df.index
        return result
        
    def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        new_col = kwargs['new_col']
        depend_on = kwargs['depend_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")
        
        tmpdf = df.reset_index()
        if not self.thinking:
            prompt = prompts.column_one_call_prompt.format(col_names=depend_on, 
                                                        col_values = tmpdf[['index'] + depend_on].to_json(orient='records'), 
                                                        condition=condition, 
                                                        new_col_name = new_col)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.column_one_call_think_prompt.format(col_names=depend_on, 
                                                        col_values = tmpdf[['index'] + depend_on].to_json(orient='records'), 
                                                        condition=condition, 
                                                        new_col_name = new_col)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']          
        result = pd.DataFrame(result)
        # print(result.shape)
        for _, row in result.iterrows():
            df.at[int(row['index']), new_col] = row[new_col]
        # df[new_col] = result
        return df

    def _row_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        example_num = kwargs['example_num']
        if example_num > 0:
            sampled_df = df[-example_num:]
            rows_prompt = ""
            for _, _r in sampled_df.iterrows():
                rows_prompt += json.dumps(_r.to_dict()) + "\n"
            extra_info = prompts.example_prompt.format(rows=rows_prompt)
        else:
            extra_info = ""

        if not self.thinking:
            prompt = prompts.row_insert_prompt.format(col_names = df.columns.values, 
                                                        description = condition, 
                                                        extra_prompt = extra_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.row_insert_think_prompt.format(col_names = df.columns.values, 
                                                        description = condition, 
                                                        extra_prompt = extra_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        df = pd.concat([df, pd.DataFrame(result)])
        return df

    def _table_execute(self, condition: str, **kwargs):
        col_names = kwargs['col_names']
        if not self.thinking:
            prompt = prompts.table_create_prompt.format(col_names = col_names, 
                                                        description = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.table_create_think_prompt.format(col_names = col_names, 
                                                        description = condition)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]            
        return pd.DataFrame(result)
            

class LLMOneImpute(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONE, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.CELL:
            return asyncio.run(self._cell_execute(condition, df, **kwargs))
        elif self.operand_type == OperandType.COLUMN:
            return asyncio.run(self._column_execute(condition, df, **kwargs))
        elif self.operand_type == OperandType.ROW:
            raise ValueError(f"Implementation type '{self.implementation_type}' does not support the operand type '{self.operand_type}'.")
            return self._row_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            raise ValueError(f"Implementation type '{self.implementation_type}' does not support the operand type '{self.operand_type}'.")
            return self._table_execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")

    async def _cell_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        example_num = kwargs['example_num']

        semaphore = asyncio.Semaphore(10)
        async def process_row(idx, row):
            async with semaphore:
                if any(pd.isna(value) for value in row):  
                    row_json = json.dumps(row.to_dict())
                    if example_num > 0:
                        df_non_null = df.dropna()
                        cur_idx = idx
                        non_null_idx = df_non_null.index.sort_values()

                        before_idx = non_null_idx[non_null_idx < cur_idx]
                        after_idx = non_null_idx[non_null_idx > cur_idx]

                        window_idx = before_idx[-example_num:] if len(before_idx) >= example_num else after_idx[:example_num]
                        df_window = df_non_null.loc[window_idx]

                        rows_prompt = ""
                        for _, _r in df_window.iterrows():
                            rows_prompt += json.dumps(_r.to_dict()) + "\n"
                        extra_info = prompts.example_prompt.format(rows=rows_prompt)
                    else:
                        extra_info = ""

                    if not self.thinking:
                        prompt = prompts.cell_impute_prompt.format(
                            row=row_json, condition=condition, extra_prompt=extra_info
                        )
                        result = await self.client.async_call([{'role': 'user', 'content': prompt}])
                        imputed_row = json_response_postprocess(result)
                    else:
                        prompt = prompts.cell_impute_think_prompt.format(
                            row=row_json, condition=condition, extra_prompt=extra_info
                        )
                        result = await self.client.async_call([{'role': 'user', 'content': prompt}])
                        result = json_response_postprocess(result)[0]
                        # print(result)
                        imputed_row = result['result']

                    # 更新缺失值
                    for column in row.index:
                        if pd.isna(row[column]):
                            row[column] = imputed_row[0][column]

                return row

        tasks = [process_row(idx, row) for idx, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)
    
    async def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        new_col = kwargs['new_col']
        depend_on = kwargs['depend_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")

        semaphore = asyncio.Semaphore(10)
        async def process_row(idx, row):
            async with semaphore: 
                row_json = {col: str(row[col]) for col in depend_on}
                row_json[new_col] = None
                row_json = json.dumps(row_json)

                if not self.thinking:
                    prompt = prompts.cell_impute_prompt.format(
                        row=row_json,
                        condition=condition,
                        extra_prompt=""
                    )
                    result = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    imputed_row = json_response_postprocess(result)
                else:
                    prompt = prompts.cell_impute_think_prompt.format(
                        row=row_json,
                        condition=condition,
                        extra_prompt=""
                    )
                    result = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    result = json_response_postprocess(result)[0]
                    # print(result)
                    imputed_row = result['result']
                df.at[idx, new_col] = imputed_row[0][new_col]


        tasks = [process_row(idx, row) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)
        return df