import re
import json
import pandas as pd 


def json_response_postprocess(response):
    """
    Parses the response from LLM and transforms it into a JSON.
    
    Args:
    - response (str): The response from the LLM, which might contain additional text.
    
    Returns:
    - JSON: The parsed data in JSON formatting.
    """

    # Use a regular expression to extract the JSON part from the response

    try:
        pattern = r"```(?:json\s+)?(.*?)```"
        response = re.sub(pattern, r'\1', response, flags=re.DOTALL)

        json_match = re.search(r'\[\s*{.*?}\s*\]', response, re.DOTALL)
        
        if json_match:
            if json_match.end() == len(response):
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
            else:
                json_data = json.loads(response)
        else:
            list_match = re.search(r'\[\s*.*?\s*\]', response, re.DOTALL)
            if list_match:
                list_str = list_match.group(0)
                json_data = json.loads(list_str)
        return json_data
    except:
        print(f"Failed response:{response}")


def df_columns_to_json(df, columns, include_keys=True):
    """
    Convert specified columns of a DataFrame to a JSON string,
    with each row as a separate JSON object or array.

    Parameters:
    df : pd.DataFrame
        The source DataFrame.
    columns : str or list
        Column name(s) to include in the JSON output.
    include_keys : bool, default True
        Whether to include column names (keys) in the JSON output.
        - True: output JSON objects, e.g. {"col1": val1, ...}
        - False: output JSON arrays of values, e.g. [val1, val2, ...]

    Returns:
    str
        JSON string with each row as a separate JSON object or array,
        separated by newline characters.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    records = df[columns].to_dict(orient='records')
    
    if include_keys:
        # Output JSON objects (dicts)
        lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    else:
        # Output lists of values without keys
        lines = [json.dumps([rec[col] for col in columns], ensure_ascii=False) for rec in records]
    
    return '\n'.join(lines)


def list_wrapper(str_array):
    joined_str =  ','.join(f'{{{item}}}' for item in str_array)
    joined_str = '[' + joined_str + ']'
    return joined_str
    