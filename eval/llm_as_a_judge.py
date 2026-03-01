import os
import sys
import csv
import time
import asyncio
import pandas as pd
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.utils.LLMCaller import LLMCaller
from src.prompts.llm_as_a_judge_prompts import LLM_AS_A_JUDGE_PROMPTS

class LLMJudgeEvaluator:
    """
    A class for evaluating predictions using LLM-as-a-judge.
    """
    
    def __init__(self, result_dir='eval/results', 
                 baselines=None, output_path='eval/results/llm_judgement_summary.csv'):
        """
        Initialize the LLM Judge Evaluator.
        
        Args:
            result_dir: Directory containing Excel files with predictions and ground truth
            baselines: List of baseline names to evaluate. If None, will use default list.
            output_path: Path to save the summary results CSV file
        """
        self.llmcaller = LLMCaller()
        
        # Search for all xlsx files in result_dir
        self.result_path_list = []
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if file.endswith('.xlsx'):
                    file_path = os.path.join(result_dir, file)
                    self.result_path_list.append(file_path)
            self.result_path_list.sort() 
        
        if not self.result_path_list:
            raise ValueError(f"No xlsx files found in {result_dir}")
        
        print(f"Found {len(self.result_path_list)} xlsx file(s) in {result_dir}:")
        for path in self.result_path_list:
            print(f"  - {path}")
        
        self.output_path = output_path
        
        if baselines is None:
            self.baselines = [
                'lotus_wo_cot', 'lotus_w_cot', 'blendsql', 
                'hqdl', 'binder', 'suql', 'palimpzest', 'thalamusdb', 'aryn', 'docetl', 'bt'
            ]
        else:
            self.baselines = baselines
    
    async def compare_with_llm_judge(self, index: int, query: str, prediction: str, ground_truth: str) -> int:
        """
        Use LLM-as-a-judge to compare prediction and ground_truth.
        
        Args:
            index: Index of the current record
            query: The user query
            prediction: The baseline prediction to evaluate
            ground_truth: The ground truth answer
            
        Returns:
            1 if accept, 0 if reject
        """
        prompt = LLM_AS_A_JUDGE_PROMPTS.format(
            query=query, 
            prediction=prediction, 
            ground_truth=ground_truth
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert on judging the equivalence of predictions and ground truth."},
                {"role": "user", "content": prompt}
            ]
            
            response_content = await self.llmcaller.async_call(messages)
            print(response_content)
            response_content =response_content.replace("```json", "").replace("```", "").strip()
            response_content = json.loads(response_content)
            judgment = response_content['judgment'].strip().upper()
            
            print("=" * 40)
            print(f"Judging Query {index} ...")
            print(f"Prediction: {prediction}")
            print(f"Ground Truth: {ground_truth}")
            print(f"LLM Judgment: {judgment}")
            
            if judgment == "ACCEPT":
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error comparing prediction and ground truth for query {index}: {e}")
            return 0
    
    def _extract_base_model(self, file_path: str) -> str:
        """
        Extract base model name from file path.
        """
        filename = os.path.basename(file_path)
        base_model = os.path.splitext(filename)[0]
        return base_model
    
    async def run(self):
        """
        Run the end-to-end evaluation pipeline
        """
        start_time = time.time()
        all_results = {}  # Store results for all base models: {base_model: {baseline: stats}}
        
        # Process each file independently (each file represents one base model)
        for result_path in self.result_path_list:
            base_model = self._extract_base_model(result_path)
            print("\n" + "=" * 80)
            print(f"Processing Base Model: {base_model}")
            print("=" * 80)
            
            # Load data for this base model
            df = pd.read_excel(result_path, engine='openpyxl')
            print(f"Reading records from: {result_path}")
            print(f"  Total rows: {len(df)}\n")
            
            # Evaluate all baselines for this base model
            base_model_results = {}
            
            for baseline in self.baselines:
                prediction_col = f'{baseline}_prediction'
                
                if prediction_col not in df.columns:
                    print(f"Warning: Column '{prediction_col}' not found. Skipping {baseline}.")
                    continue
                
                print("=" * 60)
                print(f"Evaluating baseline: {baseline} (Base Model: {base_model})")
                print(f"  Total queries to evaluate: {len(df)}")
                
                # Create tasks for parallel execution
                tasks = []
                for idx, row in df.iterrows():
                    query = row['query']
                    prediction = row[prediction_col]
                    ground_truth = row['ground_truth']
                    
                    task = self.compare_with_llm_judge(idx, query, str(prediction), str(ground_truth))
                    tasks.append(task)
                
                # Execute all tasks in parallel
                print(f"  Executing {len(tasks)} LLM judge calls in parallel...")
                match_results = await asyncio.gather(*tasks)
                
                judgement_col = f'{base_model}_{baseline}_llm_judgement'
                df[judgement_col] = match_results
                
                matches = sum(match_results)
                total = len(match_results)
                match_rate = matches / total if total > 0 else 0
                
                result = {
                    'matches': matches,
                    'total': total,
                    'match_rate': match_rate,
                }
                base_model_results[baseline] = result
                
                print(f"\n{baseline} Results (Base Model: {base_model}):")
                print(f"  Matches: {matches}/{total}")
                print(f"  Match rate: {match_rate:.2%}\n")
            
            # Save results back to the original file
            df.to_excel(result_path, index=False, engine='openpyxl')
            print(f"Results saved back to: {result_path}")
            
            all_results[base_model] = base_model_results
        
        # Print summary results for all base models
        print("\n" + "=" * 80)
        print("Summary of All Base Models and Baselines")
        print("=" * 80)
        
        for base_model, base_model_results in all_results.items():
            print(f"\nBase Model: {base_model}")
            print("-" * 60)
            for baseline, stats in base_model_results.items():
                print(f"  {baseline:20s}: {stats['matches']:3d}/{stats['total']:3d} ({stats['match_rate']:6.2%})")
        
        end_time = time.time()
        print(f"\nTime taken to evaluate all base models: {end_time - start_time:.2f} seconds")
        
        # Save summary results to CSV
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['base_model', 'baseline', 'matches', 'total', 'match_rate'])
            for base_model, base_model_results in all_results.items():
                for baseline, stats in base_model_results.items():
                    writer.writerow([
                        base_model,
                        baseline, 
                        stats['matches'], 
                        stats['total'], 
                        stats['match_rate']
                    ])
        print(f"\nSummary results saved to: {self.output_path}")


def main():
    """Main function to run the evaluator."""
    evaluator = LLMJudgeEvaluator()
    asyncio.run(evaluator.run())


if __name__ == "__main__":
    main()
