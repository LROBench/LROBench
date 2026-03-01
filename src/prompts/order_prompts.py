rows_one_call_prompt = '''
Task
Given multiple rows in a table, rank the top '{k}' rows in '{ascending}' order based on the '{metric}' criteria of the  '{depend_on}' column.
Input
- A JSON-formatted table with multiple rows
{rows}

Instructions
1. Carefully evaluate the '{metric}' criteria of the '{depend_on}' column in each row.
2. Select and rank the top '{k}' items:
When ranking by 'ascending' order:
- Rank 1 = its '{metric}' metric value of the '{depend_on}' column is the lowest 
- Rank '{k}' = its '{metric}' metric value of the '{depend_on}' colum is the '{k}'-th lowest
When ranking by 'descending' order:
- Rank 1 = its '{metric}' metric value of the '{depend_on}' column is the highest
- Rank '{k}' = its '{metric}' metric value of the '{depend_on}' column is the '{k}'-th highest 
3. If multiple items seem equally qualified, use your best judgment to break ties
4. Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary
5. Do not modify any original values in the rows, including the index. 

Output
Return ONLY a JSON format table with exactly '{k}' objects in ranked order (1 to '{k}'), using this format:
[
  {{"index":"{{index}}", "{{col_1}}": "value", ..., "{{col_n}}": "value"}},
  ...
  {{"index":"{{index}}", "{{col_1}}": "value", ..., "{{col_n}}": "value"}}
]

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''

pair_judge_prompt = '''
Task
Given two items, compare their '{metric}' criteria of the '{depend_on}' column

Inputs
- Item 1 
{row1}
- Item 2
{row2}

Instructions
- Carefully compare the '{metric}' criteria of the '{depend_on}' column
- Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary

Output
Return ONLY a single character 1 or 0, where 
- 1 indicates that the Item 1 has a higher '{metric}' metric value than Item 2
- 0 indicates that the Item 2 has a higher '{metric}' metric value than Item 1

Ensure that the comparison of two rows fully depends on their '{metric}' criteria of the '{depend_on}' column
Return only one single character 1 or 0 without any additional text, formatting, code blocks, or explanations.
'''

row_scoring_prompt = '''
Task
Given an item '{row}', assign a score between 0 and 100 to evaluate its '{metric}' metric value of the '{depend_on}' column

Inputs
- Item 
{row}

Instructions
- Carefully evaluate the '{metric}' criteria of the '{depend_on}' column
- A higher score indicates that the '{depend_on}' column value has a higher '{metric}' metric. Conversely, a lower score indicates a lower metric.
- Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary

Output Format
Return ONLY an interger between 0 and 100 in string format

Ensure that the score precisely reflects the '{metric}' criteria of the '{depend_on}' column
Return only an interger between 0 and 100 in string format without any additional text, formatting, code blocks, or explanations.
'''



rows_one_call_think_prompt = '''
Task
Given multiple rows in a table, rank the top '{k}' rows in '{ascending}' order based on the '{metric}' criteria of the  '{depend_on}' column.
Input
- A JSON-formatted table with multiple rows
{rows}

Instructions
1. Carefully evaluate the '{metric}' criteria of the '{depend_on}' column in each row.
2. Select and rank the top '{k}' items:
When ranking by 'ascending' order:
- Rank 1 = its '{metric}' metric value of the '{depend_on}' column is the lowest 
- Rank '{k}' = its '{metric}' metric value of the '{depend_on}' colum is the '{k}'-th lowest
When ranking by 'descending' order:
- Rank 1 = its '{metric}' metric value of the '{depend_on}' column is the highest
- Rank '{k}' = its '{metric}' metric value of the '{depend_on}' column is the '{k}'-th highest 
3. If multiple items seem equally qualified, use your best judgment to break ties
4. Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary
5. Do not modify any original values in the rows, including the index. 

Output Format
Return the output using the following JSON format. 
To enhance the result, as briefly as possible include your thinking process in the JSON structure:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"index":"{{index}}", "{{col_1}}": "value", ..., "{{col_n}}": "value"}},
      ...
      {{"index":"{{index}}", "{{col_1}}": "value", ..., "{{col_n}}": "value"}}
    ]
  }}
]

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''

pair_judge_think_prompt = '''
Task
Given two items, compare their '{metric}' criteria of the '{depend_on}' column

Inputs
- Item 1 
{row1}
- Item 2
{row2}

Instructions
- Carefully compare the '{metric}' criteria of the '{depend_on}' column
- Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary

Output Format
Return the output using the following JSON format. 
To enhance the result, as briefly as possible include your thinking process in the JSON structure:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{0/1 value}}"
  }}
]
where the result value must be an integer '0' or '1'. 
- 1 indicates that the Item 1 has a higher metric value than Item 2
- 0 indicates that the Item 1 has a smaller metric value than Item 2

Ensure that the comparison of two rows fully depends on their '{metric}' criteria of the '{depend_on}' column
Return the ouput strictly in the given format without any extra word outside JSON structure.
'''

row_scoring_think_prompt = '''
Task
Given an item '{row}', assign a score between 0 and 100 to evaluate its '{metric}' metric value of the '{depend_on}' column

Inputs
- Item 
{row}

Instructions
- Carefully evaluate the '{metric}' criteria of the '{depend_on}' column
- A higher score indicates that the '{depend_on}' column value has a higher '{metric}' metric. Conversely, a lower score indicates a lower metric.
- Use external knowledge to infer the '{metric}' criteria based on the '{depend_on}' column when necessary

Output Format
Return the output using the following JSON format. 
To enhance the result, as briefly as possible include your thinking process in the JSON structure:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{0-100 score}}"
  }}
]
where the result value must be an integer ranging from 0 to 100.

Ensure that the score precisely reflects the '{metric}' criteria of the '{depend_on}' column
Return the ouput strictly in the given format without any extra word outside JSON structure.
'''