cells_one_call_prompt = '''
Task: Given a table with multiple rows, some rows have missing cell values. 
You need to reasonably infer and impute the values of all the missing cell values in this table, according to the '{condition}' condition.

Inputs:
- A table with missing cell values
{tab}

Instructions:
- Infer the value of each missing cell by utilizing the information from its corresponding row whenever possible.
- If inferring from the row is not possible, you must use external knowledge to impute the value of the missing cell

Output Format:
Return an imputed version of the input table in the following JSON format:
[
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}},
  ...
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


column_one_call_prompt = '''
Task: Given a list of columns '{col_names}', reasonably infer and impute the values of a new column '{new_col_name}' according to the '{condition}' condition.

Inputs:
- A list of columns '{col_names}', and the values of these columns are as follows:
{col_values}

Instructions
- Infer the value of the new column by utilizing the information from the given columns whenever possible.
- Use external knowledge to impute the new column if necessary.
- The imputed column must contain exactly the same number of rows as the input columns.
- The output table must contain corresponding index column.

Output Format:
Return all the values of imputed column in the following JSON format:
[
  {{"index":"index", "{new_col_name}": "{{new_col_value}}"}},
  ...
  {{"index":"index", "{new_col_name}": "{{new_col_value}}"}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


cell_impute_prompt = '''
Task: Given a missing cell in a JSON formatted row '{row}', reasonably infer or impute this cell according to the '{condition}' condition.

Inputs:
- A JSON formatted row with missing cells
'{row}'

{extra_prompt}

Instructions:
- If example rows are given above, carefully refer to their data format or specific values when necessary.
- Infer the value of each missing cell by utilizing the information from its corresponding row whenever possible.
- If inferring from the row is not possible, you must use external knowledge to impute the value of the missing cell.

Output Format:
Return the imputed row in the following JSON format referring to the '{condition}' condition:
[
  {{"{{col_1}}": "value", 
  ..., 
  "{{col_n}}": "value"}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


example_prompt = '''
Here are some examples of the neighboring complete rows from this table. Carefully refer to their data format or specific values when necessary.
{rows}
'''


row_insert_prompt = '''
Task
Create a single new row for the table whose columns are:
'{col_names}'

Inputs
- Column names: '{col_names}'
- Description: '{description}'

{extra_prompt}

Instructions
- Read the description and fill in the corresponding column values.
- If the description omits certain details, use well-known facts or reliable public knowledge to supply them.
- If example rows are provided, match their data format exactly (do not copy their values).

Output Format:
Return the new row in the following JSON format:
[
  {{"{{col_1}}": "value", 
  ..., 
  "{{col_n}}": "value"}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


table_create_prompt = '''
Task
Create a new table—containing as many rows as possible—based on the given description.
The table must use the exact column names provided in '{col_names}'.
Leverage external knowledge as much as possible.

Inputs
- Column names: '{col_names}'
- Description: '{description}'

Instructions
- Include every column listed in '{col_names}' in each row.
- Add as many valid rows (tuples) as possible referring to the '{description}' description.
- Populate each cell with accurate, sensible values; infer or research details when the description is incomplete.

Output format
Return the table in the following JSON format:
[
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value"}},
  ...,
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value"}}
]
Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


cells_one_call_think_prompt = '''
Task: Given a table with multiple rows, some rows have missing cell values. 
You need to reasonably infer and impute the values of all the missing cell values in this table, according to the '{condition}' condition.

Inputs:
- A table with missing cell values
{tab}

Instructions:
- Infer the value of each missing cell by utilizing the information from its corresponding row whenever possible.
- If inferring from the row is not possible, you must use external knowledge to impute the value of the missing cell

Output Format:
Return the output using the following JSON format. 
To enhance the result, briefly include your thinking or reasoning process within the JSON structure:
[
  {{
    "thinking": "{{...}}", 
    "result":
    [
      {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}},
      ...
      {{"{{col_1}}": "value", ..., "{{col_n}}": "value",}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''



column_one_call_think_prompt = '''
Task: Given a list of columns '{col_names}', reasonably infer and impute the values of a new column '{new_col_name}' according to the '{condition}' condition.

Inputs:
- A list of columns '{col_names}', and the values of these columns are as follows:
{col_values}

Instructions
- Infer the value of the new column by utilizing the information from the given columns whenever possible.
- Use external knowledge to impute the new column if necessary.
- The imputed column must contain exactly the same number of rows as the input columns.
- The output table must contain corresponding index column.

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result":
    [
      {{"index":"index", "{new_col_name}": "{{new_col_value}}"}},
      ...
      {{"index":"index", "{new_col_name}": "{{new_col_value}}"}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''



cell_impute_think_prompt = '''
Task: Given a missing cell in a row '{row}', reasonably infer or impute this cell according to the '{condition}' condition.

Inputs:
'{row}'

{extra_prompt}

Instructions:
- If example rows are given above, carefully refer to their data format or specific values when necessary.
- Infer the value of each missing cell by utilizing the information from its corresponding row whenever possible.
- If inferring from the row is not possible, you must use external knowledge to impute the value of the missing cell.

Output Format:
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result":
    [
      {{"{{col_1}}": "value",  ..., "{{col_n}}": "value"}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''



row_insert_think_prompt = '''
Task
Create a single new row for the table whose columns are:
'{col_names}'

Inputs
- Column names: '{col_names}'
- Description: '{description}'

{extra_prompt}

Instructions
- Read the description and fill in the corresponding column values.
- If the description omits certain details, use well-known facts or reliable public knowledge to supply them.
- If example rows are provided, match their data format exactly (do not copy their values).

Output Format:
Return the new row in the following JSON format.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result":
      [
        {{"{{col_1}}": "value", 
        ..., 
        "{{col_n}}": "value"}}
      ]
  }}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


table_create_think_prompt = '''
Task
Create a new table—containing as many rows as possible—based on the given description.
The table must use the exact column names provided in '{col_names}'.
Leverage external knowledge as much as possible.

Inputs
- Column names: '{col_names}'
- Description: '{description}'

Instructions
- Include every column listed in '{col_names}' in each row.
- Add as many valid rows (tuples) as possible referring to the '{description}' description.
- Populate each cell with accurate, sensible values; infer or research details when the description is incomplete.

Output format
Return the table in the following JSON format:
[
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value"}},
  ...,
  {{"{{col_1}}": "value", ..., "{{col_n}}": "value"}}
]
Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''