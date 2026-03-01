# rows_one_call_prompt = '''
# Task: Given a table, group all the rows into distinct clusters based on the '{depend_on}' column, and the clustering should satisfy the '{condition}' condition.

# Inputs:
# - A table in JSON format
# {tab}

# Instructions:
# - Carefully cluster all the rows in this table into distinct clusters based on the '{depend_on}' column.
# - All the clusters must satisfy the '{condition}' condition.
# - Carefully summarize a concise cluster name for each cluster, based on the values of the '{depend_on}' column and the '{condition}' condition.
# - Each row must be assigned to exactly one cluster.
# - After clustering, apply the '{agg}' aggregation operation on the '{agg_on}' column to aggregate the data of each cluster.

# Output Format: 
# Return a table in the following JSON format:
# [
#   {{ "cluster_name": "cluster_name_1", "{agg}_result": "value", }}, 
#   ...,
#   {{ "cluster_name": "cluster_name_k", "{agg}_result": "value",}}, 
# ]
# If the '{agg}' aggregation operation name is 'DISTINCT', only return the "cluster" column, but do not return the '{agg}'_result values!

# Return only the JSON content without any additional text, formatting, code blocks, or explanations.
# '''

rows_one_call_prompt = '''
Task: Given a table, group all the rows into distinct clusters based on the '{depend_on}' column, and the clustering should satisfy the '{condition}' condition.

Inputs:
- A table in JSON format
{tab}

Instructions:
- Carefully cluster all the rows in this table into distinct clusters based on the '{depend_on}' column.
- All the clusters must satisfy the '{condition}' condition. Each cluster with a unique cluster label.
- Carefully summarize a concise cluster name for each cluster, based on the values of the '{depend_on}' column and the '{condition}' condition.
- Each row must be assigned to exactly one cluster. The rows in the output should exactly match the input rows and correspond one-to-one.
- The output table must contain exactly the same number of rows as the input table.
- The output table must contain corresponding index column.

Output Format: 
Return a list of cluster labels and cluster name for each row in the following JSON format:
[
  {{"index":"{{index}}", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}, 
  ...,
  {{"index":"{{index}}", "cluster_name": "{{cluster_name}}",  "cluster_label": "{{label}}"}}
]
Note that "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


row_judge_prompt = '''
Task: Given a string, assign it to an existing cluster or a new cluster based on the '{depend_on}' column.
The clustering must satisfy the '{condition}' condition.

Inputs:
- A row in JSON format
{row}
- Existing cluster labels and names
{clusters_prompt}

Instructions:
- Carefully assign the row to an existing cluster or a new cluster, based on the '{depend_on}' column and the '{condition}' condition.
- Carefully design a concise cluster name for each cluster, satisfying the '{condition}' condition.
- The row must be assigned to exactly one cluster.

Output Format: 
Return the cluster label of this row in the following JSON format:
[
  {{"index":"{{index}}", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}
]
Note that "{{row_index}}" and "{{cluster_label}}" are integers in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''


columns_one_call_prompt = '''
Task: Given the columns in a table, group all these columns into distinct clusters, and the clustering should satisfy the '{condition}' condition.

Inputs:
- The columns in a table
{cols}

{extra_prompt}

Instructions:
- Carefully cluster all the columns into distinct clusters.
- All the clusters must satisfy the '{condition}' condition.
- Carefully summarize a concise cluster name for each cluster, based on the column names in this cluster and the '{condition}' condition.
- Each cluster must be mutually exclusive (no overlaps), and the union of all clusters must include all columns.
- The output order of columns should be the same as the input order.

Output Format: 
Return a table in the following JSON format:
[
  {{ "column_name": "column_1", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}, 
  ...,
  {{ "column_name": "column_n", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}
]
Note that "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''

column_example_prompt = '''
Here are some examples of the column values {indicator}:
- Column names:
{col_names}
- First few rows of the columns:
{col_values}
'''

col_tab_judge_prompt = '''
Task: Given a column or a table '{string}', assign it to an existing cluster or a new cluster based on the '{condition}' condition.

Inputs:
- The name of a column or a table
{string}
- Existing cluster labels and names
{clusters_prompt}

{extra_prompt}

Instructions:
- Carefully assign this column (or table) name '{string}' to an existing cluster or a new cluster based on the '{condition}' condition.
- Carefully design a concise cluster name for each cluster based on the '{condition}' condition. A
- The column (or table) name must be assigned to exactly one cluster.

Output Format: 
Return the cluster label of this column (or table) in the following JSON format:
[
  {{ "{string}": ["{{cluster_label}}", "{{cluster_name}}"]}}
]
Note that "{{cluster_label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''

tables_one_call_prompt = '''
Task: Given the tables '{tables}' in a database, group all these tables into distinct clusters, and the clustering should satisfy the '{condition}' condition.

Inputs:
- A list of tables
{table_cols}

{extra_prompt}

Instructions:
- Carefully cluster all the tables into distinct clusters.
- All the clusters must satisfy the '{condition}' condition.
- Carefully summarize a concise cluster name for each cluster, based on the table names in this cluster and the '{condition}' condition.
- Each cluster must be mutually exclusive (no overlaps), and the union of all clusters must include all tables.
- The output order of tables should be the same as the input order.

Output Format: 
Return a table in the following JSON format:
[
  {{ "table_name": "table_1", "cluster_name": "cluster_name", "cluster_label": "{{label}}"}}, 
  ...,
  {{ "table_name": "table_n", "cluster_name": "cluster_name", "cluster_label": "{{label}}"}} 
]
Note that "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return only the JSON content without any additional text, formatting, code blocks, or explanations.
'''



rows_one_call_think_prompt = '''
Task: Given a table, group all the rows into distinct clusters based on the '{depend_on}' column, and the clustering should satisfy the '{condition}' condition.

Inputs:
- A table in JSON format
{tab}

Instructions:
- Carefully cluster all the rows in this table into distinct clusters based on the '{depend_on}' column.
- All the clusters must satisfy the '{condition}' condition. Each cluster with a unique cluster label.
- Carefully summarize a concise cluster name for each cluster, based on the values of the '{depend_on}' column and the '{condition}' condition.
- Each row must be assigned to exactly one cluster. The rows in the output should exactly match the input rows and correspond one-to-one.
- The output table must contain exactly the same number of rows as the input table.
- The output table must contain corresponding index column.

Output Format: 
Return a list of cluster labels and cluster name for each row in the following JSON format.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"index":"{{index}}", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}, 
      ...,
      {{"index":"{{index}}", "cluster_name": "{{cluster_name}}",  "cluster_label": "{{label}}"}}
    ]
  }}
]
where "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


row_judge_think_prompt = '''
Task: Given a string, assign it to an existing cluster or a new cluster based on the '{depend_on}' column.
The clustering must satisfy the '{condition}' condition.

Inputs:
- A row in JSON format
{row}
- Existing cluster labels and names
{clusters_prompt}

Instructions:
- Carefully assign the row to an existing cluster or a new cluster, based on the '{depend_on}' column and the '{condition}' condition.
- Carefully design a concise cluster name for each cluster, satisfying the '{condition}' condition.
- The row must be assigned to exactly one cluster.

Output Format: 
Return the cluster label of this row in the following JSON format:
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{"index":"{{index}}", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}
    ]
  }}
]
where "{{row_index}}" and "{{cluster_label}}" are integers in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


columns_one_call_think_prompt = '''
Task: Given the columns in a table, group all these columns into distinct clusters, and the clustering should satisfy the '{condition}' condition.

Inputs:
- The columns in a table
{cols}

{extra_prompt}

Instructions:
- Carefully cluster all the columns into distinct clusters.
- All the clusters must satisfy the '{condition}' condition.
- Carefully summarize a concise cluster name for each cluster, based on the column names in this cluster and the '{condition}' condition.
- Each cluster must be mutually exclusive (no overlaps), and the union of all clusters must include all columns.
- The output order of columns should be the same as the input order.

Output Format: 
Return a table using the following JSON format.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{ "column_name": "column_1", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}}, 
      ...,
      {{ "column_name": "column_n", "cluster_name": "{{cluster_name}}", "cluster_label": "{{label}}"}} 
    ]
  }}
]
where "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''


col_tab_judge_think_prompt = '''
Task: Given a column or a table '{string}', assign it to an existing cluster or a new cluster based on the '{condition}' condition.

Inputs:
- The name of a column or a table
{string}
- Existing cluster labels and names
{clusters_prompt}

{extra_prompt}

Instructions:
- Carefully assign this column (or table) name '{string}' to an existing cluster or a new cluster based on the '{condition}' condition.
- Carefully design a concise cluster name for each cluster based on the '{condition}' condition. A
- The column (or table) name must be assigned to exactly one cluster.

Output Format: 
Return the cluster label of this column (or table) in the following JSON format.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{ "{string}": ["{{cluster_label}}", "{{cluster_name}}"]}}
    ]
  }}
]
where "{{cluster_label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''

tables_one_call_think_prompt = '''
Task: Given the tables '{tables}' in a database, group all these tables into distinct clusters, and the clustering should satisfy the '{condition}' condition.

Inputs:
- A list of tables
{table_cols}

{extra_prompt}

Instructions:
- Carefully cluster all the tables into distinct clusters.
- All the clusters must satisfy the '{condition}' condition.
- Carefully summarize a concise cluster name for each cluster, based on the table names in this cluster and the '{condition}' condition.
- Each cluster must be mutually exclusive (no overlaps), and the union of all clusters must include all tables.
- The output order of tables should be the same as the input order.

Output Format: 
Return a table in the following JSON format.
To enhance the result, include your thinking process within the JSON as brief as possible:
[
  {{
    "thinking": "{{...}}", 
    "result": 
    [
      {{ "table_name": "table_1", "cluster_name": "cluster_name", "cluster_label": "{{label}}"}}, 
      ...,
      {{ "table_name": "table_n", "cluster_name": "cluster_name", "cluster_label": "{{label}}"}}
    ]
  }}
]
where "{{label}}" is an integer in string format, and "{{cluster_name}}" is a highly concise and brief cluster name.

Return the ouput strictly in the given format without any extra word outside JSON structure.
'''