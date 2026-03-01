from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.order_prompts as prompts
import pandas as pd
import json
from src.utils.parse import json_response_postprocess
from src.utils.heap import MinHeap, MaxHeap
import asyncio

class LLMAllOrder(PhysicalOperator):
    def __init__(self, operand_type: OperandType,thinking: bool = False):
        super().__init__(ImplType.LLM_ALL, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return self._row_execute(condition, df, depend_on, k, ascending, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _row_execute(self, condition: str, df: pd.DataFrame, depend_on: list|str, k: int, ascending: bool, **kwargs) -> pd.DataFrame:
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")

        tmpdf = df[depend_on].reset_index()
        if not self.thinking:
            prompt = prompts.rows_one_call_prompt.format(rows = tmpdf.to_json(orient='records'), 
                                                        metric = condition, 
                                                        depend_on = depend_on, 
                                                        ascending = 'ascending' if ascending else 'descending', 
                                                        k = k)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            print(result)
            result = json_response_postprocess(result)
        else:
            prompt = prompts.rows_one_call_think_prompt.format(rows = tmpdf.to_json(orient='records'), 
                                                        metric = condition, 
                                                        depend_on = depend_on, 
                                                        ascending = 'ascending' if ascending else 'descending', 
                                                        k = k)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            print(result)
            result = json_response_postprocess(result)[0]
            result = result['result']
        new_oreder = pd.DataFrame(result)['index'].astype(int)
        df_reordered = df.loc[new_oreder]
        return df_reordered
        

class LLMOneOrder(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONE, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return asyncio.run(self._row_execute(condition, df, depend_on, ascending, k, **kwargs))
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    async def _row_execute(self, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs) -> pd.DataFrame:
        def descend_compare(row1, row2):
            # return 1 when row1 > row2 
            if not self.thinking:
                prompt = prompts.pair_judge_prompt.format(row1=row1, row2=row2, depend_on=depend_on, metric=condition)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
            else:
                prompt = prompts.pair_judge_think_prompt.format(row1=row1, row2=row2, depend_on=depend_on, metric=condition)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
                flag = json_response_postprocess(flag)[0]
                print(flag)
                flag = flag['result']
            return int(flag)
        def ascend_compare(row1, row2):
            # return 1 when row1 < row2 
            if not self.thinking:
                prompt = prompts.pair_judge_prompt.format(row1=row2, row2=row1, depend_on=depend_on, metric=condition)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
            else:
                prompt = prompts.pair_judge_think_prompt.format(row1=row2, row2=row1, depend_on=depend_on, metric=condition)
                flag = self.client.call([{'role': 'user', 'content': prompt}])
                flag = json_response_postprocess(flag)[0]
                print(flag)
                flag = flag['result']
            return int(flag)
        
        sort_algo = kwargs['sort_algo']

        if sort_algo == 'heap':
            tmpdf = df[depend_on].reset_index()
            rows = tmpdf.to_dict(orient='records')
            if ascending == False:
                heap = MinHeap(k = k, cmp = descend_compare)
            else:
                heap = MaxHeap(k = k, cmp = ascend_compare)
            for row in rows:
                heap.add(row)
            res = heap.get_topk()
            res = pd.DataFrame(res)
            sorted_idx = res['index']
            return df.loc[sorted_idx]
        elif sort_algo == 'simple':
            tmpdf = df[depend_on].reset_index()
            rows = tmpdf.to_dict(orient='records')
            semaphore = asyncio.Semaphore(10)
            async def process_pair(row1, row2):
                async with semaphore:
                    if not self.thinking:
                        prompt = prompts.pair_judge_prompt.format(row1=row1, row2=row2, depend_on=depend_on, metric=condition)
                        flag = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    else:
                        prompt = prompts.pair_judge_think_prompt.format(row1=row1, row2=row2, depend_on=depend_on, metric=condition)
                        flag = await self.client.async_call([{'role': 'user', 'content': prompt}])
                        flag = json_response_postprocess(flag)[0]
                        # print(flag)
                        flag = flag['result']
                    print(flag)
                    return int(flag)


            tasks = [process_pair(row1, row2) for row1 in rows for row2 in rows if row1 != row2]
            results = await asyncio.gather(*tasks)

            counts = [0] * len(rows)
            idx = 0
            for i, row1 in enumerate(rows):
                for j, row2 in enumerate(rows):
                    if i != j:  # 非自身
                        counts[i] += results[idx]
                        idx += 1

            df['tmp_rank'] = counts
            df = df.sort_values(by='tmp_rank', ascending=ascending)
            df = df.drop('tmp_rank', axis=1)
            return df[:k]
                    

class LLMSemiOrder(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_SEMI, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return asyncio.run(self._row_execute(condition, df, depend_on, ascending, k, **kwargs))
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
    
    async def _row_execute(self, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs) -> pd.DataFrame:
        tmpdf = df[depend_on].reset_index()
        rows = tmpdf.to_dict(orient='records')

        semaphore = asyncio.Semaphore(10)
        async def process_row(row):
            async with semaphore:
                if not self.thinking:
                    prompt = prompts.row_scoring_prompt.format(row = json.dumps(row), 
                                                                depend_on = depend_on, 
                                                                metric = condition)
                    score = await self.client.async_call([{'role': 'user', 'content': prompt}])
                else:
                    prompt = prompts.row_scoring_think_prompt.format(row = json.dumps(row), 
                                                                depend_on = depend_on, 
                                                                metric = condition)
                    score = await self.client.async_call([{'role': 'user', 'content': prompt}])
                    score = json_response_postprocess(score)[0]
                    print(score)
                    score = score['result']
                    if not isinstance(score, str):
                        return score
                return float(score.strip('"').strip('\''))

        tasks = [process_row(row) for row in rows]
        scores = await asyncio.gather(*tasks)
        df['score'] = scores

        df = df.sort_values(by='score', ascending=ascending)
        return df[:k].reset_index()