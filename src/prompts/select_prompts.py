rows_one_call_prompt = '''
Task: Given a table, select a subset of rows which '{depend_on}' columns satisfy the '{condition}' condition.

Inputs:
- A JSON-formatted table
{tab}

Instructions:
- Fully leverage the input data information to make accurate judgments.
- Use external knowledge when necessary.
- The output table must contain corresponding index column.

Output Format:
Return a subset of the input table in the following JSON format, including only the rows which '{depend_on}' columns satisfy the '{condition}' condition:
[
  {{"{{index}}": "value", "{{col_1}}": "value", ..., "{{col_n}}": "value",}},
  ...
  {{"{{index}}": "value", "{{col_1}}": "value", ..., "{{col_n}}": "value",}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


columns_one_call_prompt = '''
Task: Given a lists of columns, select a subset of columns which satisfy the '{condition}' condition.

Inputs:
- Columns in JSON format 
{col_list}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Use external knowledge when necessary.

Output Format:
Return a subset of the input columns, including only the columns satisfying the '{condition}' condition:
[
  "col_1", ..., "col_n"
]

These columns should be those that fully satisfy the '{condition}' condition. 
Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''




tables_one_call_prompt = '''
Task: Given a lists of tables '{tables}', select a subset of tables which satisfy the '{condition}' condition.

Inputs:
{table_cols}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Use external knowledge when necessary.

Output Format:
Return a subset of tables, including only the tables satisfying the '{condition}' condition:
[
  "table_1", 
  ..., 
  "table_n"
]

These tables should be those that fully satisfy the '{condition}' condition. 
Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''

row_one_judge_prompt = '''
Task: Given a tuple '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information to make accurate judgments.
- Use external knowledge when necessary.

Output Format: 
Strictly return an integer '0' or '1'. 
'0' means the input does not satisfy the condition, and '1' means the input satisfies the condition.

Return only one character without any additional text, formatting, code blocks, or explanations. 
'''

column_one_judge_prompt = '''
Task: Given a column named '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information to make accurate judgments.
- Use external knowledge when necessary.

Output Format: 
Strictly return an integer '0' or '1'. 
'0' means the input does not satisfy the condition, and '1' means the input satisfies the condition.

Return only one character without any additional text, formatting, code blocks, or explanations. 
'''



table_one_judge_prompt = '''
Task: Given a table named '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information to make accurate judgments.
- Use external knowledge when necessary.

Output Format: 
Strictly return an integer '0' or '1'. 
'0' means the input does not satisfy the condition, and '1' means the input satisfies the condition.

Return only one character without any additional text, formatting, code blocks, or explanations. 
'''

column_example_prompt = '''
Here are some examples of the column values {indicator}:
- Column names:
{column_names}
- First few rows of the columns:
{column_values}
'''




rows_one_call_think_prompt = '''
Task: Given a table, select a subset of rows which '{depend_on}' columns satisfy the '{condition}' condition.

Inputs:
{tab}

Instructions:
- Fully leverage the input data information to make accurate judgments.
- Use external knowledge when necessary.
- The output table must contain corresponding index column.

Output Format:
Return the output using the following JSON format. 
If a row is selected, output the entire row in "result".
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
      [
        {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}},
        ...,
        {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}}
      ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''






columns_one_call_think_prompt = '''
Task: Given a lists of column names, select a subset of columns which satisfy the '{condition}' condition.

Inputs:
{col_list}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Use external knowledge when necessary.

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result":
      [
        "col_1", ..., "col_k"
      ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
Return the "result" strictly in the given format without adding extra brackets around the column name.
'''




tables_one_call_think_prompt = '''
Task: Given a lists of tables '{tables}', select a subset of tables which satisfy the '{condition}' condition.

Inputs:
{table_cols}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Use external knowledge when necessary.

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result":
      [
        "table_1", ..., "table_k"
      ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''



row_one_judge_think_prompt = '''
Task: Given a tuple '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information and use external knowledge when necessary.

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



column_one_judge_think_prompt = '''
Task: Given a column named '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information and use external knowledge when necessary.

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


table_one_judge_think_prompt = '''
Task: Given a table named '{row}', determine whether it satisfies the '{condition}' condition.

Inputs:
{row}

{extra_prompt}

Instructions:
- If examples are given above, carefully refer to their data format or specific values when necessary.
- Fully leverage the input data information and use external knowledge when necessary.

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
