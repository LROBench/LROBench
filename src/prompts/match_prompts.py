cells_one_call_prompt = '''
Task: Given two lists of distinct values from columns '{col_l}' and '{col_r}', determine which pairs satisfy the '{condition}' condition.

Inputs:
- Column Left: '{col_l}' with values {col_l_val}
- Column Right: '{col_r}' with values {col_r_val}

Instructions:
- Evaluate each possible pair (from values in Column Left and Column Right) based on the '{condition}' condition.
- Identify all pairs where the combination of values from Column Left and Column Right meet this condition.

Output Format:
Return the results in the following JSON format, including only the pairs that satisfy the '{condition}' condition:
[
  {{"{col_l}": "value 1", "{col_r}": "value 3",}},
  ...
  {{"{col_l}": "value 2", "{col_r}": "value 4",}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


rows_one_call_prompt = '''
Task: Given two tables, Table Left and Table Right, perform entity matching based on the '{condition}' condition.
One entity matching pair consists of one row from Table Left and one row from Table Right, and they must satisfy the '{condition}' condition.

Inputs:
- Table Left values in json format 
{tab_l_val}
- Table Right values in json format
{tab_r_val}

Instructions:
- Evaluate each possible entity matching row pair (from rows in Table Left and Table Right).
- Identify all row pairs meeting the '{condition}' condition.

Output Format:
Return the matched pairs in the following JSON format, where each row represents a merged row combining one row from Table Left and its matched row from Table Right:
[
  {{"{{left_column_name_1}}": "{{value}}", ... , "{{left_column_name_x}}": "{{value}}", "{{right_column_name_1}}": "{{value}}", ..., "{{right_column_name_y}}": "{{value}}",}},
  ...
  {{"{{left_column_name_1}}": "{{value}}", ... , "{{left_column_name_x}}": "{{value}}", "{{right_column_name_1}}": "{{value}}", ..., "{{right_column_name_y}}": "{{value}}",}}
]

Return only the JSON itself without code block, without formatting, without explanations. 
'''


columns_one_call_prompt = '''
Task: Given two lists of column names respectively from Table Left and Table Right, determine which pairs satisfy the '{condition}' condition.

Inputs:
- Columns from Table Left in JSON format 
{col_list_l}
- Columns from Table Right in JSON format 
{col_list_r}

{extra_prompt}

Instructions:
- Evaluate each possible entity matching column pair (from columns in Table Left and Table Right).
- Identify all column pairs meeting the '{condition}' condition.

Output Format:
Return the results in the following JSON format, including only the column name pairs that satisfy the '{condition}' condition:
[
  {{"Table Left": "left_column_name", "Table Right": "right_column_name",}},
  ...
  {{"Table Left": "left_column_name", "Table Right": "right_column_name",}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''

column_example_prompt = '''
Here are some examples of the column values {indicator}:
- Column names:
{column_names}
- First few rows of the columns:
{column_values}
'''


pair_judge_prompt = '''
Task: Given two string values or two JSON tuples '{val_l}' and '{val_r}', determine whether they satisfy the '{condition}' condition.

Inputs:
- Two string values or JSON tuples:
{val_l}
{val_r}

{extra_prompt}

Output Format: 
Strictly return an integer '0' or '1'. 

Return only one character without any additional text, formatting, code blocks, or explanations. 
'''


cell_choice_judge_prompt = '''
Match the value '{row}' from the '{col_l}' column to the corresponding value in the '{col_r}' column, referring to the '{condition}' condition.

The list of available values in '{col_r}' are as follows:
{val_list_r}

Strictly return only one matched value satisfying the '{condition}' condition, without code block, without formatting, without explanations. 
If no matched value exists, return an empty string.
'''


row_choice_judge_prompt = '''
Match the JSON-formatted left row '{row}' to the corresponding right row in the given rows below, referring to the '{condition}' condition.

The right optional rows are given in JSON format as follows:
{options}

Output Format:
Return the matched row in the following JSON format, which satisfies the '{condition}' condition.
[
  {{"{{left_col_1}}": "{{value}}", ... , "{{left_col_m}}": "{{value}}", "{{right_col_1}}": "{{value}}"  ..., "{{right_col_n}}": "{{value}}"}}
]
  
If no matched row exists, return an empty JSON.
Strictly return only one matched row in JSON format, without any additional text, formatting, code blocks, or explanations. 
'''


column_choice_judge_prompt = '''
Match the column '{item}' in Table Left to the correspnding column from columns in Table Right below, referring to the '{condition}' condition.

The names of the optional columns in Table Right are as follows:
{options}

{extra_prompt}

Strictly return only one matched column in pure string format satisfying the '{condition}' condition, without any additional text, formatting, code blocks, or explanations. 
If no matched column exists, return an empty string.
'''


cells_one_call_think_prompt = '''
Task: Given two lists of distinct values from columns '{col_l}' and '{col_r}', determine which pairs satisfy the '{condition}' condition.

Inputs:
- Column Left: '{col_l}' with values {col_l_val}
- Column Right: '{col_r}' with values {col_r_val}

Instructions:
- Evaluate each possible pair (from values in Column Left and Column Right) based on the '{condition}' condition.
- Identify all pairs where the combination of values from Column Left and Column Right meet this condition.

Output Format:
Return the results in the following JSON format, including only the pairs that satisfy the condition.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"{col_l}": "value 1", "{col_r}": "value 3",}},
      ...
      {{"{col_l}": "value 2", "{col_r}": "value 4",}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


rows_one_call_think_prompt = '''
Task: Given two tables, Table Left and Table Right, perform entity matching based on the '{condition}' condition.
One entity matching pair consists of one row from Table Left and one row from Table Right, and they must satisfy the '{condition}' condition.

Inputs:
- Table Left values in json format 
{tab_l_val}
- Table Right values in json format
{tab_r_val}

Instructions:
- Evaluate each possible entity matching row pair (from rows in Table Left and Table Right).
- Identify all row pairs meeting the '{condition}' condition.

Output Format:
Return the matched pairs in the following JSON format, where each row represents a merged row combining one row from Table Left and its matched row from Table Right.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"{{left_column_name_1}}": "{{value}}", ... , "{{left_column_name_x}}": "{{value}}", "{{right_column_name_1}}": "{{value}}", ..., "{{right_column_name_y}}": "{{value}}",}},
      ...
      {{"{{left_column_name_1}}": "{{value}}", ... , "{{left_column_name_x}}": "{{value}}", "{{right_column_name_1}}": "{{value}}", ..., "{{right_column_name_y}}": "{{value}}",}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


columns_one_call_think_prompt = '''
Task: Given two lists of column names respectively from Table Left and Table Right, determine which pairs satisfy the '{condition}' condition.

Inputs:
- Columns from Table Left in JSON format 
{col_list_l}
- Columns from Table Right in JSON format 
{col_list_r}

{extra_prompt}

Instructions:
- Evaluate each possible entity matching column pair (from columns in Table Left and Table Right).
- Identify all column pairs meeting the '{condition}' condition.

Output Format:
Return the results in the following JSON format, including only the column name pairs that satisfy the '{condition}' condition:
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"Table Left": "left_column_name", "Table Right": "right_column_name",}},
      ...
      {{"Table Left": "left_column_name", "Table Right": "right_column_name",}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''



pair_judge_think_prompt = '''
Task: Given two string values or two JSON tuples '{val_l}' and '{val_r}', determine whether they satisfy the '{condition}' condition.

Inputs:
- Two string values or JSON tuples:
{val_l}
{val_r}

{extra_prompt}

Output Format: 
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{0/1 value}}"
  }}
]
where the result value must be an integer '0' or '1'. 
'0' means the input does not satisfy the condition, and '1' means the input satisfies the condition.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


cell_choice_judge_think_prompt = '''
Task: Match the value '{row}' from the '{col_l}' column to the corresponding value in the '{col_r}' column, referring to the '{condition}' condition.

Input: The list of available values in '{col_r}' are as follows:
{val_list_r}

Output Format: 
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{one matched value}}"
  }}
]
Strictly return only one matched value satisfying the condition in the "result".
If no matched value exists, return an empty string.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


row_choice_judge_think_prompt = '''
Match the JSON-formatted left row '{row}' to the corresponding right row in the given rows below, referring to the '{condition}' condition.

The right optional rows are given in JSON format as follows:
{options}

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"{{left_col_1}}": "{{value}}", ... , "{{left_col_m}}": "{{value}}", "{{right_col_1}}": "{{value}}"  ..., "{{right_col_n}}": "{{value}}"}}
    ]
  }}
]
Strictly return only one matched row satisfying the condition in the "result".
If no matched row exists, return an empty list in the "result".

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


column_choice_judge_think_prompt = '''
Task: Match the column '{item}' in Table Left to the correspnding column from columns in Table Right below, referring to the '{condition}' condition.

Inputs: The names of the optional columns in Table Right are as follows:
{options}

{extra_prompt}

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{one matched column name}}"
  }}
]
Strictly return only one matched column name satisfying the condition in the "result".
If no matched column name exists, return an empty string.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''