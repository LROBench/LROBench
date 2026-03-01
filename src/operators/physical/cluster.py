from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.cluster_prompts as prompts
import pandas as pd
import json
from src.utils.parse import json_response_postprocess, list_wrapper

class LLMAllCluster(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ALL, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return self._row_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _row_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        depend_on = kwargs['depend_on']
        # agg = kwargs['agg']
        # agg_on = kwargs['agg_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")
            
        # if agg.lower() not in ['count', 'mean', 'sum', 'min', 'max', 'distinct']:
        #     raise ValueError("Unsupported agg function type.")
        # if not isinstance(agg_on, str):
        #     raise ValueError("The parameter and 'agg_on' must be a string.")
        tmpdf = df[depend_on].reset_index()
        if not self.thinking:
            prompt = prompts.rows_one_call_prompt.format(condition = condition, 
                                                tab = tmpdf.to_json(orient='records'),
                                                depend_on = depend_on)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.rows_one_call_think_prompt.format(condition = condition, 
                                                tab = tmpdf.to_json(orient='records'),
                                                depend_on = depend_on)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        result = pd.DataFrame(result)

        for _, row in result.iterrows():
            df.at[int(row['index']), 'cluster_label'] = int(row['cluster_label'])
            df.at[int(row['index']), 'cluster_name'] = row['cluster_name']
        
        maxv = int(df['cluster_label'].max())
        null_indices = df[df['cluster_label'].isnull()].index
        for i, idx in enumerate(null_indices):
            df.loc[idx, 'cluster_label'] = maxv + 1 + i
            df.loc[idx, 'cluster_name'] = str(maxv + 1 + i) + ' unique cluster'
        df['cluster_label'] = df['cluster_label'].astype(int)
        
        return df
    
    def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        example_num = kwargs['example_num']
        columns = df.columns.values

        extra_prompt = ""
        if example_num > 0:
            for col in columns:
                extra_prompt += prompts.column_example_prompt.format(col_names = col, 
                                                                     col_values = df[[col]][:example_num].to_json(),
                                                                     indicator = f"in the column '{col}'")
                extra_prompt += "\n"
        if not self.thinking:
            prompt = prompts.columns_one_call_prompt.format(condition = condition, 
                                                            cols = list_wrapper(columns),
                                                            extra_prompt=extra_prompt)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.columns_one_call_think_prompt.format(condition = condition, 
                                                            cols = list_wrapper(columns),
                                                            extra_prompt=extra_prompt)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        result = pd.DataFrame(result)
        result['cluster_label'] = result['cluster_label'].map(int)
        return result

    
    def _table_execute(self, condition: str, df: list, **kwargs):
        table_names = kwargs['table_names']
        example_num = kwargs['example_num']

        table_col_prompt = "- Input: \n"
        for (table, table_df) in zip(table_names, df):
            _cols = list_wrapper(table_df.columns)
            _prompt = f"Table '{table}' have columns '{_cols}'"
            table_col_prompt += _prompt + "\n"
        
        example_info = ""
        if example_num > 0:
            for (table, table_df) in zip(table_names, df):
                _prompt = prompts.column_example_prompt.format(col_names = list_wrapper(table_df.columns), 
                                                                col_values = table_df[:example_num].to_json(orient='records'),
                                                                indicator = f"in Table '{table}'")
                example_info += _prompt + "\n"
        if not self.thinking:
            prompt = prompts.tables_one_call_prompt.format(tables=table_names, 
                                                condition=condition, 
                                                table_cols=table_col_prompt, 
                                                extra_prompt=example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)
        else:
            prompt = prompts.tables_one_call_think_prompt.format(tables=table_names, 
                                                condition=condition, 
                                                table_cols=table_col_prompt, 
                                                extra_prompt=example_info)
            result = self.client.call([{'role': 'user', 'content': prompt}])
            result = json_response_postprocess(result)[0]
            # print(result)
            result = result['result']
        result = pd.DataFrame(result)
        result['cluster_label'] = result['cluster_label'].map(int)
        return result

    
class LLMOneCluster(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONE, operand_type, thinking)

    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> pd.DataFrame:
        if self.operand_type == OperandType.ROW:
            return self._row_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.COLUMN:
            return self._column_execute(condition, df, **kwargs)
        elif self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _row_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        depend_on = kwargs['depend_on']
        # agg = kwargs['agg']
        # agg_on = kwargs['agg_on']
        if isinstance(depend_on, str):
            depend_on = [depend_on]
        elif isinstance(depend_on, list):
            pass
        else:
            raise ValueError("The parameter 'depend_on' must be a column name or a list of column names.")
        # if not (isinstance(agg, str) and isinstance(agg_on, str)):
        #     raise ValueError("The parameters 'agg' and 'agg_on' must be a string.")
        
        tmpdf = df[depend_on].reset_index()
        rows = tmpdf.to_dict(orient='records')
        clusters = {}
        cluster_names = []
        cluster_labels = []
        results = []
        for row in rows:
            # row_json = {col: row[col] for col in depend_on}
            if not self.thinking:
                prompt = prompts.row_judge_prompt.format(condition = condition, 
                                                        row = row, 
                                                        depend_on = depend_on, 
                                                        clusters_prompt=json.dumps(clusters))
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = dict(json_response_postprocess(result)[0]) # result: {'row_id': xx, 'cluster_id': xx, 'cluster_name': xx]}
                results.append(result)
            else:
                prompt = prompts.row_judge_think_prompt.format(condition = condition, 
                                                        row = row, 
                                                        depend_on = depend_on, 
                                                        clusters_prompt=json.dumps(clusters))
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = json_response_postprocess(result)[0]
                # print(result)
                result = dict(result['result'][0])
                results.append(result)
            if result['cluster_label'] not in clusters:
                clusters[result['cluster_label']] = result['cluster_name'] # clusters: {cluster_id: cluster_name}

        # df['cluster_name'] = cluster_names
        # df['cluster_label'] = cluster_labels
        results = pd.DataFrame(results)
        for _, row in results.iterrows():
            df.at[int(row['index']), 'cluster_label'] = int(row['cluster_label'])
            df.at[int(row['index']), 'cluster_name'] = row['cluster_name']
        df['cluster_label'] = df['cluster_label'].map(int)
        return df


    def _column_execute(self, condition: str, df: pd.DataFrame, **kwargs):
        example_num = kwargs['example_num']
        columns = df.columns.values

        clusters = {}
        cluster_names = []
        cluster_labels = []
        for col in columns:
            extra_prompt = ""
            if example_num > 0:
                extra_prompt = prompts.column_example_prompt.format(col_names = col, 
                                                                    col_values = df[[col]][:example_num].to_json(),
                                                                    indicator = f"in the column '{col}'")
            if not self.thinking:
                prompt = prompts.col_tab_judge_prompt.format(string = col, 
                                                            condition = condition, 
                                                            clusters_prompt = json.dumps(clusters), 
                                                            extra_prompt = extra_prompt)
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = json_response_postprocess(result)[0]
            else:
                prompt = prompts.col_tab_judge_think_prompt.format(string = col, 
                                                            condition = condition, 
                                                            clusters_prompt = json.dumps(clusters), 
                                                            extra_prompt = extra_prompt)
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = json_response_postprocess(result)[0]
                # print(result)
                result = result['result'][0]

            for val in result.values():
                cluster_labels.append(val[0].strip("\"").strip("\'"))
                cluster_names.append(val[1])
                if val[0] not in clusters:
                    clusters[val[0]] = val[1] # clusters: {cluster_id: cluster_name}

        output = {"column_name": columns, "cluster_name": cluster_names, "cluster_label": cluster_labels}
        output = pd.DataFrame(output)
        output["cluster_label"] = output["cluster_label"].map(int)
        return output

    def _table_execute(self, condition: str, df: list, **kwargs):
        table_names = kwargs['table_names']
        example_num = kwargs['example_num']

        clusters = {}
        cluster_names = []
        cluster_labels = []
        for (table, table_df) in zip(table_names, df):
            extra_prompt = ""
            if example_num > 0:
                extra_prompt = prompts.column_example_prompt.format(col_names = table_df.columns.values, 
                                                                    col_values =table_df[:example_num].to_json(orient='records'),
                                                                    indicator = f"in Table '{table}'")
            if not self.thinking:
                prompt = prompts.col_tab_judge_prompt.format(string = table, 
                                                            condition = condition, 
                                                            clusters_prompt = json.dumps(clusters), 
                                                            extra_prompt = extra_prompt)
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = json_response_postprocess(result)[0]
            else:
                prompt = prompts.col_tab_judge_think_prompt.format(string = table, 
                                                            condition = condition, 
                                                            clusters_prompt = json.dumps(clusters), 
                                                            extra_prompt = extra_prompt)
                result = self.client.call([{'role': 'user', 'content': prompt}])
                result = json_response_postprocess(result)[0]
                # print(result)
                result = result['result'][0]

            for val in result.values():
                cluster_labels.append(val[0].strip("\"").strip("\'"))
                cluster_names.append(val[1])
                if val[0] not in clusters:
                    clusters[val[0]] = val[1] # clusters: {cluster_id: cluster_name}

        output = {"table_name": table_names, "cluster_name": cluster_names, "cluster_label": cluster_labels}
        output = pd.DataFrame(output)
        output["cluster_label"] = output["cluster_label"].map(int)
        return output