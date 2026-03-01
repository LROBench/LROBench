from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.match_prompts as prompts
import pandas as pd
import json
from src.utils.parse import json_response_postprocess, df_columns_to_json, list_wrapper
import asyncio

class LLMAllMatch(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ALL, operand_type, thinking)

    def execute(self, condition:str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.CELL: 
            return self._cell_execute(condition, left_df, right_df, **kwargs)
        elif self.operand_type == OperandType.ROW:
            return self._row_execute(condition, left_df, right_df, **kwargs)
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, left_df, right_df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")

    def _cell_execute(self, condition:str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)
        left_on = 'left_' + kwargs['left_on']
        right_on = 'right_' + kwargs['right_on']

        left_val = left_df[left_on].drop_duplicates().values
        left_val = json.dumps(left_val.tolist())
        right_val = right_df[right_on].drop_duplicates().values
        right_val = json.dumps(right_val.tolist())

        if not self.thinking:
            prompt = prompts.cells_one_call_prompt.format(col_l = left_on, 
                                                        col_r = right_on, 
                                                        col_l_val = left_val, 
                                                        col_r_val = right_val,  
                                                        condition = condition) 
            # print(prompt)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            middle_df = pd.DataFrame(json_response_postprocess(result))
        else:
            prompt = prompts.cells_one_call_think_prompt.format(col_l = left_on, 
                                                        col_r = right_on, 
                                                        col_l_val = left_val, 
                                                        col_r_val = right_val,  
                                                        condition = condition) 
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            middle_df = pd.DataFrame(result['result'])
        if len(middle_df) == 0:
            return pd.DataFrame()
        final_df = pd.merge(left_df, middle_df, left_on=left_on, right_on=left_on)
        final_df = pd.merge(final_df, right_df, left_on=right_on, right_on=right_on)
        
        return final_df
                
    def _row_execute(self, condition:str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)
        
        if not self.thinking:
            prompt = prompts.rows_one_call_prompt.format(tab_l_val = left_df.to_json(orient='records'),
                                                            tab_r_val = right_df.to_json(orient='records'),
                                                            condition = condition)
            
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.rows_one_call_think_prompt.format(tab_l_val = left_df.to_json(orient='records'),
                                                            tab_r_val = right_df.to_json(orient='records'),
                                                            condition = condition)
            
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        return pd.DataFrame(result)
    
    def _column_execute(self, condition:str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        example_num = kwargs['example_num']
        if example_num == 0:
            example_info = ""
        else:
            left_prompt = prompts.column_example_prompt.format(column_names = list_wrapper(left_df.columns), 
                                                                    column_values = left_df[:example_num].to_json(),
                                                                    indicator = "in Table Left")
            right_prompt = prompts.column_example_prompt.format(column_names = list_wrapper(right_df.columns), 
                                                                    column_values = right_df[:example_num].to_json(),
                                                                    indicator = "in Table Right")
            example_info = left_prompt + right_prompt

        if not self.thinking:
            prompt = prompts.columns_one_call_prompt.format(col_list_l = list_wrapper(left_df.columns), 
                                                                col_list_r =  list_wrapper(right_df.columns), 
                                                                condition = condition, 
                                                                extra_prompt = example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.columns_one_call_think_prompt.format(col_list_l = list_wrapper(left_df.columns), 
                                                                col_list_r =  list_wrapper(right_df.columns), 
                                                                condition = condition, 
                                                                extra_prompt = example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        return pd.DataFrame(result)

    
class LLMOneMap(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONE, operand_type, thinking)

    async def pair_judge(self, val_l, val_r, condition, extra_prompt = ""):
        if not self.thinking:
            prompt = prompts.pair_judge_prompt.format(val_l=val_l, val_r=val_r, condition=condition, extra_prompt=extra_prompt)
            result = await self.client.async_call([{'role': 'user', 'content': prompt}])
        else:
            prompt = prompts.pair_judge_think_prompt.format(val_l=val_l, val_r=val_r, condition=condition, extra_prompt=extra_prompt)
            result = await self.client.async_call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            print(result)
            result = result['result']
        return int(result)

    def execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.CELL: 
            return asyncio.run(self._cell_execute(condition, left_df, right_df, **kwargs))
        elif self.operand_type == OperandType.ROW:
            return asyncio.run(self._row_execute(condition, left_df, right_df, **kwargs))
        elif self.operand_type == OperandType.COLUMN:
            return asyncio.run(self._column_execute(condition, left_df, right_df, **kwargs))
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")

    async def _cell_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)
        left_on = 'left_' + kwargs['left_on']
        right_on = 'right_' + kwargs['right_on']


        full_cross_df = pd.merge(left_df[[left_on]].drop_duplicates(), 
                                 right_df[[right_on]].drop_duplicates(), how='cross')

        semaphore = asyncio.Semaphore(10)
        async def process_row(index, row):
            async with semaphore:
                # print(f"Task started: index={index}, semaphore available={semaphore._value}")
                flag = await self.pair_judge(row[left_on], row[right_on], condition)
                return index, flag
        async def process_all_rows():
            tasks = [process_row(index, row) for index, row in full_cross_df.iterrows()]
            return await asyncio.gather(*tasks)
        
        results = await process_all_rows()
        flags = [0] * len(full_cross_df)
        for index, flag in results:
            flags[index] = flag
        
        full_cross_df['flag'] = flags 
        mid_df = full_cross_df[full_cross_df['flag'] == 1] 
        mid_df = mid_df.drop('flag', axis=1) 
        if len(mid_df) == 0:
            return pd.DataFrame()
        
        final_df = pd.merge(left_df, mid_df, on=left_on)
        final_df = pd.merge(final_df, right_df, on=right_on)
        return final_df 
        
    async def _row_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)
            
        semaphore = asyncio.Semaphore(10)  
        async def process_row(index, row):
            async with semaphore:
                left_row_json = json.dumps(row[left_df.columns].to_dict())
                right_row_json = json.dumps(row[right_df.columns].to_dict())
                flag = await self.pair_judge(left_row_json, right_row_json, condition)
                await asyncio.sleep(1)
                return index, flag
        full_cross_df = pd.merge(left_df, right_df, how='cross')
        tasks = [process_row(idx, row) for idx, row in full_cross_df.iterrows()]
        results = await asyncio.gather(*tasks)
        flags = [0] * len(full_cross_df)
        for index, flag in results:
            flags[index] = flag
        full_cross_df['flag'] = flags
        final_df = full_cross_df[full_cross_df['flag'] == 1]
        final_df = final_df.drop('flag', axis=1)
        return final_df
    
    async def _column_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        example_num = kwargs['example_num']

        semaphore = asyncio.Semaphore(10)  
        async def compare_columns(l_col, r_col):
            async with semaphore:  
                if example_num == 0:
                    res = await self.pair_judge(l_col, r_col, condition)
                    if res:
                        return {"Table Left": l_col, "Table Right": r_col}
                else:
                    left_prompt = prompts.column_example_prompt.format(
                        column_names=l_col,
                        column_values=left_df[[l_col]][:example_num].to_json(),
                        indicator="in column '" + l_col + "'"
                    )
                    right_prompt = prompts.column_example_prompt.format(
                        column_names=r_col,
                        column_values=right_df[[r_col]][:example_num].to_json(),
                        indicator="in column '" + r_col + "'"
                    )
                    res = await self.pair_judge(val_l=l_col, val_r=r_col, condition=condition, extra_prompt=left_prompt + right_prompt)
                    if res:
                        return {"Table Left": l_col, "Table Right": r_col}
                return None 
        tasks = [
            compare_columns(l_col, r_col)
            for l_col in left_df.columns for r_col in right_df.columns
        ]
        results = await asyncio.gather(*tasks)
        results = [result for result in results if result is not None]
        return pd.DataFrame(results)


class LLMSemiMap(PhysicalOperator): 
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_SEMI, operand_type, thinking)

    def execute(self, condition:str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.CELL: 
            return asyncio.run(self._cell_execute(condition, left_df, right_df, **kwargs))
        elif self.operand_type == OperandType.ROW:
            return asyncio.run(self._row_execute(condition, left_df, right_df, **kwargs))
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, left_df, right_df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
    
    async def _cell_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)
        left_on = 'left_' + kwargs['left_on']
        right_on = 'right_' + kwargs['right_on']

        options = df_columns_to_json(right_df, right_on, include_keys=False)
        questions = left_df[left_on].drop_duplicates().values

        semaphore = asyncio.Semaphore(10) 

        async def process_question(question):
            async with semaphore:  
                if not self.thinking:
                    prompt = prompts.cell_choice_judge_prompt.format(
                        row=question, val_list_r=options, col_l=left_on, col_r=right_on, condition=condition
                    )
                    answer = await self.client.async_call([{'role': 'user', 'content': prompt}])
                else:
                    prompt = prompts.cell_choice_judge_think_prompt.format(
                        row=question, val_list_r=options, col_l=left_on, col_r=right_on, condition=condition
                    )
                    answer = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    answer = json_response_postprocess(answer)[0]
                    # print(answer)
                    answer = answer['result']
                if answer != "":
                    return {left_on: question, right_on: answer}
                return None

        tasks = [process_question(question) for question in questions]
        raw_results = await asyncio.gather(*tasks)
        results = [res for res in raw_results if res is not None]

        mid_df = pd.DataFrame(results)
        if len(mid_df) == 0:
            return pd.DataFrame()
        mid_df[left_on] = mid_df[left_on].astype(left_df[left_on].dtype)
        mid_df[right_on] = mid_df[right_on].astype(right_df[right_on].dtype)

        final_df = pd.merge(left_df, mid_df, left_on=left_on, right_on=left_on)
        final_df = pd.merge(final_df, right_df, left_on=right_on, right_on=right_on)

        return final_df
        
    async def _row_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for col in left_df.columns.values:
            left_df.rename(columns={col: 'left_'+col}, inplace=True)
        for col in right_df.columns.values:
            right_df.rename(columns={col: 'right_'+col}, inplace=True)

        options = right_df.to_json(orient='records')
        semaphore = asyncio.Semaphore(10)
        async def process_row(row):
            async with semaphore:
                _row = json.dumps(row.to_dict())
                if not self.thinking:
                    prompt = prompts.row_choice_judge_prompt.format(row=_row, options=options, condition=condition)
                    answer = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    answer = json_response_postprocess(answer)
                else:
                    prompt = prompts.row_choice_judge_think_prompt.format(row=_row, options=options, condition=condition)
                    answer = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    answer = json_response_postprocess(answer)[0]
                    # print(answer)
                    answer = answer['result']
                if len(answer) > 0:
                    return answer[0]
                else:
                    return None
        
        tasks = [process_row(row) for _, row in left_df.iterrows()]
        results = await asyncio.gather(*tasks)
        results = [row for row in results if row is not None]
        return pd.DataFrame(results)
        
    def _column_execute(self, condition: str, left_df: pd.DataFrame, right_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        example_num = kwargs['example_num']

        options = list_wrapper(right_df.columns.values)
        results = []
        for question in left_df.columns.values:
            extra_prompt = ""
            if example_num > 0:
                item_prompt = prompts.column_example_prompt.format(column_names=question, 
                                                                        column_values=left_df[[question]][:example_num].to_json(),
                                                                        indicator="in Table Left")
                option_prompt = prompts.column_example_prompt.format(column_names=options,
                                                                            column_values=right_df[:example_num].to_json(),
                                                                            indicator="in Table Right")
                extra_prompt = item_prompt + option_prompt
            if not self.thinking:
                prompt = prompts.column_choice_judge_prompt.format(item=question, options=options, 
                                                                    condition=condition, 
                                                                    extra_prompt=extra_prompt)
                answer = self.client.call([{'role': 'user', 'content': prompt}])
            else:
                prompt = prompts.column_choice_judge_think_prompt.format(item=question, options=options, 
                                                                    condition=condition, 
                                                                    extra_prompt=extra_prompt)
                answer = self.client.call([{'role': 'user', 'content': prompt}])
                answer = json_response_postprocess(answer)[0]
                # print(answer)
                answer = answer['result']
            if len(answer.replace('\'', '').replace('\"', '')) > 0:
                results.append({"Table Left": question, "Table Right": answer})
        return pd.DataFrame(results)

