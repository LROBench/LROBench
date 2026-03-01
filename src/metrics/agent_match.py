from src.utils.LLMCaller import LLMCaller
import pandas as pd 
from src.prompts.score_prompts import string_score_prompt
import asyncio
from src.utils.parse import json_response_postprocess

async def agent_match(str_a, str_b):
    client = LLMCaller()
    prompt = string_score_prompt.format(str_a=str_a, str_b=str_b)
    response = await client.async_call([{'role': 'user', 'content': prompt}])
    response = json_response_postprocess(response)[0]
    # print(response)
    response = int(response['result'])
    return response

def df_agent_acc(repaired_df: pd.DataFrame, missing_log: list) -> float:
    return asyncio.run(async_df_agent_acc(repaired_df, missing_log))

async def async_df_agent_acc(repaired_df, missing_log) -> float:
    semaphore = asyncio.Semaphore(10)

    async def process_row(row_idx, col, true_val):
        pred_val = repaired_df.at[row_idx, col]
        if pd.isna(pred_val):
            return
        async with semaphore:
            return await agent_match(pred_val, true_val)
        
    tasks = [process_row(row_idx, col, true_val) for row_idx, col, true_val in missing_log]
    result = await asyncio.gather(*tasks)

    correct = sum(result)
    total = len(missing_log)
    return round(correct * 1.0 / total, 4) if total else 0.0

def list_agent_acc(list_true, list_pred):
    return asyncio.run(async_list_agent_acc(list_true, list_pred))

async def async_list_agent_acc(list_true, list_pred):
    assert len(list_true) == len(list_pred)
    semaphore = asyncio.Semaphore(10)
    
    async def process_item(item_true, item_pred):
        async with semaphore:
            return await agent_match(str(item_pred), str(item_true))

    tasks = [process_item(item_true, item_pred) for item_true, item_pred in zip(list_true, list_pred)]
    result = await asyncio.gather(*tasks)
    correct = sum(result)
    return round(correct * 1.0 / len(list_true), 4)