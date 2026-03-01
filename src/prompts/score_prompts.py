string_score_prompt = '''
Task
Determine whether '{str_a}' and '{str_b}' convey the same meaning.

Instructions
- Consider synonyms, paraphrasing, and factual overlap.
- Ignore superficial differences such as punctuation, capitalization, or word order that do not affect meaning.
- If the two strings are float numbers, return 1 only if their relative error is less than 0.01.

Output Format
Return the output using the following JSON format. 
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": "{{0/1 value}}"
  }}
]
where the result value must be an integer '0' or '1'. 
'1' means they convey similar the same meaning, '0' otherwise/

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


plan_score_prompt = '''
Task
Evaluate the effectiveness of a SQL plan with UDFs in solving a natural language problem.
The final output should be a single JSON object.

Instructions
- Judge whether the SQL plan with UDFs can solve the natural lanugage problem:
0 = cannot solve, 1 = fully solve
- The order of invocation for SQL components should comply with the defined rules and effectively address the problem.
- Carefully judge whether the condition in the problem is completely equal to the SQL language. 
- The columns and tables involved in the SQL should be real in databases and free of hallucinations.
- Step-by-step break down and verificate the plan to ensure every step is correct.

Information about the SQL UDFs
{udf_info}

Information about the databases
{db_info}

Input
- Problem: {question}
- Plan: {plan}

Output Format
Output a score with brief thinking process as following.
[
  {{
    "thinking": {{...}},
    "score": "0/1 value"
  }}
]
Return the ouput strictly in the given format without any extra word outside JSON structure.
'''

db_prompt = '''
Here are the information about a table '{table}' in the database.
The table '{table}' contains the following columns: '{columns}'
The first few rows of the table are given below.
{rows}
'''

suql_prompt = '''
Here are the information about the SQL dialect {{SUQL}} used in the following plan, which provies 2 special SQL UDFs.
- Answer(t, q): Return the answer to question q on value t, where t may be a cell or a column (i.e., a cell list). If t is a column, the question q is performed on each row.
- Summary(t): Return the summary of t, where t may be a cell or a column (i.e., a cell list).
'''

