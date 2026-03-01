import argparse
from src.core.enums import ImplType, OperandType, OperatorType
from src.utils.common import append_result, args_to_filename, mean_by_position
from eval import *
from multiprocessing import Process, Manager
from os.path import join

def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def is_bad(res, tok):
    bad = {"ERROR", "TIMEOUT"}
    return any(r in bad for r in res) or any(t in bad for t in tok)

def wrap_query(id, return_dict, args):
    query_name = f"pipeline_{id}"
    query = globals().get(query_name, None)
    if query is None:
        return_dict['result'] = 'Not Implemented'
        return_dict['tokens'] = 'Not Implemented'
        print(f"pipeline_{id}" + " is not implemented")
    else:
        return_dict['result'], return_dict['tokens'] = query(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True, type=str, choices=[e.value for e in OperatorType])
    # parser.add_argument("--operand", required=True, type=str, choices=[e.value for e in OperandType])
    parser.add_argument("--impl", required=True, type=str,choices=[e.value for e in ImplType])
    parser.add_argument("--thinking",action="store_true", default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="./result/")
    parser.add_argument("--example_num", type=int, default=None)
    parser.add_argument("--sort_algo", type=str, default=None)


    args = parser.parse_args()
    args.operator = OperatorType(args.operator)
    args.impl = ImplType(args.impl) 
    print("=" * 120)
    args_string = f"Operator: {args.operator}, Impl: {args.impl}, Start: {args.start}, End: {args.end}, Thinking: {args.thinking}"
    if args.example_num is not None:
        args_string += f", Example: {args.example_num}"
    if args.sort_algo is not None:
        args_string += f", Sort_Algo: {args.sort_algo}"
    print(args_string)
    print("=" * 120)
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    
    max_try = 5

    for id in range(args.start, args.end):
        func_suffix = args.operator.value + str(id) if args.operator != OperatorType.MULTI else str(id)
        print(f"pipeline_{func_suffix}" + " starts")
        manager = Manager()
        return_dict = manager.dict()
        last_err = None


        valid_results_runs = []  
        valid_tokens_runs  = []

        for attempt in range(1, max_try + 1):
            return_dict = Manager().dict()   
            try:
                p = Process(target=wrap_query, args=(func_suffix, return_dict, args))
                p.start()
                p.join(timeout= 30 * 60)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    results = ["TIMEOUT"]
                    tokens  = ["TIMEOUT"]
                else:
                    results = to_list(return_dict.get("result", "ERROR"))
                    tokens  = to_list(return_dict.get("tokens", "ERROR"))

                if is_bad(results, tokens):
                    if attempt == max_try:
                        results, tokens  = ["ERROR"], ["ERROR"]
                    continue

                valid_results_runs.append(results)
                valid_tokens_runs.append(tokens)

                if len(valid_results_runs) >= 3:
                    break

            except Exception as e:
                if attempt == max_try:
                    results, tokens  = ["ERROR"], ["ERROR"]
                    pass
                continue

        avg_results = mean_by_position(valid_results_runs) if valid_results_runs else ["ERROR"]
        avg_tokens  = mean_by_position(valid_tokens_runs)  if valid_tokens_runs  else ["ERROR"]

        file_name = args_to_filename(args, ".csv")
        append_result(join(args.out_dir, file_name), func_suffix, avg_results, avg_tokens)



if __name__ == "__main__":
    main()

