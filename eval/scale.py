import pandas as pd 
import os 
import sys
import time
from multiprocessing import Process, Manager
import argparse
import tiktoken
from os.path import join

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.utils.common import args_to_filename, append_result
from src.operators.logical import *
from src.core.enums import OperandType, ImplType, OperatorType
from src.metrics.classic import f1_score
from src.metrics.agent_match import list_agent_acc


# ns = [4, 8, 21, 43, 86, 217, 434, 869, 2125, 4208, 8375]
ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def wrap_query(query, return_dict, args):
    if query is None:
        return_dict['time'] = 'Not Implemented'
        return_dict['tokens'] = 'Not Implemented'
        print(f"{query}" + " is not implemented")
    else:
        return_dict['results'], return_dict['time'], return_dict['tokens'] = query(args)


def vary_size_exp(operator, file_name, query, ns, args):
    timeout = False
    for n in ns:
        setattr(args, "n", n)
        manager = Manager()
        return_dict = manager.dict()

        print("=" * 120)
        print(f"Operator: {operator}, Impl: {args.impl}, Thinking: {args.thinking}, N: {args.n}")

        if timeout:
            append_result(join("./result/new_scale/", file_name), n, ['TIMEOUT'], ['TIMEOUT'])
            continue
        try:
            p = Process(target=wrap_query, args=(query, return_dict, args))
            p.start()
            p.join(timeout=40*60)
            if p.is_alive():
                p.terminate()
                results = ['TIMEOUT']
                time_cost = ['TIMEOUT']
                tokens = ['TIMEOUT']
                timeout = True
            else:
                results = return_dict.get('results', ['ERROR'])
                time_cost = [return_dict.get('time', ['ERROR'])]
                tokens = time_cost + list(return_dict.get('tokens', ['ERROR']))
        except Exception as e:
            print(e)
            results = ['ERROR']
            time_cost = ['ERROR']
            tokens = ['ERROR']
        append_result(join("./result/new_scale/", file_name), n, results, tokens)

        time.sleep(10)


def select_scale(random_seed):
    def syn_select_eval(args):
        query = "Filter all the players who are born after the fall of the Berlin Wall"
        player = pd.read_csv("./databases/european_football_2/Player.csv")
        player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        player = player[:args.n]

        truth = player[pd.to_datetime(player['birthday']) > pd.to_datetime('1989/11/9')]

        start_time = time.time()
        op = LogicalSelect(operand_type=OperandType.ROW)
        pred = op.execute(impl_type = args.impl, 
                        condition = "The player is born after the fall of the Berlin Wall",
                        df = player,
                        depend_on = ['birthday'],
                        thinking = args.thinking)
        print(pred)
        end_time = time.time()
        return f1_score(truth, pred), end_time-start_time, op.get_tokens()

    query = syn_select_eval
    operator = OperatorType.SELECT

    for impl in [ImplType.LLM_ALL, ImplType.LLM_ONE]:
        thinking = True
        args = argparse.Namespace(operator=operator, impl=impl, thinking=thinking,  
                                    start = 'syn0', end = 'syn0')
        file_name = args_to_filename(args, ".csv")
        vary_size_exp(operator, file_name, query, ns, args)


def impute_scale(random_seed):
    def syn_impute_eval(args):
        query = "Determine the age of each player as of 2025."
        player = pd.read_csv("./databases/european_football_2/Player.csv")
        player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        player = player[:args.n]

        def get_zodiac(month, day):
            zodiac_ranges = {
                'Aries': ((3, 21), (4, 19)),    
                'Taurus': ((4, 20), (5, 20)),      
                'Gemini': ((5, 21), (6, 21)),     
                'Cancer': ((6, 22), (7, 22)),    
                'Leo': ((7, 23), (8, 22)),      
                'Virgo': ((8, 23), (9, 22)),      
                'Libra': ((9, 23), (10, 23)),     
                'Scorpio': ((10, 24), (11, 22)),    
                'Sagittarius': ((11, 23), (12, 21)),   
                'Capricorn': ((12, 22), (1, 19)),    
                'Aquarius': ((1, 20), (2, 18)),    
                'Pisces': ((2, 19), (3, 20))      
            }
            for zodiac, ((start_month, start_day), (end_month, end_day)) in zodiac_ranges.items():
                if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                    return zodiac
            return None
        
        player['month'] = pd.to_datetime(player['birthday']).dt.month
        player['day'] = pd.to_datetime(player['birthday']).dt.day
        player['zodiac'] = player.apply(lambda row: get_zodiac(row['month'], row['day']), axis=1)
        truth = player['zodiac'].tolist()
        player = player.drop(['zodiac'], axis=1)

        start_time = time.time()
        op = LogicalImpute(operand_type=OperandType.COLUMN)
        pred = op.execute(impl_type = args.impl, 
                        condition = "Impute the zodiac based on the birthday",
                        df = player,
                        depend_on = ['birthday'],
                        new_col = 'zodiac',
                        thinking = args.thinking)
        print(pred)
        pred = pred['zodiac'].astype(str).tolist()
        end_time = time.time()
        return [list_agent_acc(truth, pred)], end_time-start_time, op.get_tokens()

    query = syn_impute_eval
    operator = OperatorType.IMPUTE

    for impl in [ImplType.LLM_ALL, ImplType.LLM_ONE]:
        thinking = True
        args = argparse.Namespace(operator=operator, impl=impl, thinking=thinking,  
                                    start = 'syn0', end = 'syn0')
        file_name = args_to_filename(args, ".csv")
        vary_size_exp(operator, file_name, query, ns, args)


# def groupby_scale():
#     def syn_groupby_eval(args):
#         query = "Group the birthdates by zodiacs"
#         player = pd.read_csv("./databases/european_football_2/Player.csv")
#         player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#         player = player[:args.n]

#         def get_zodiac(month, day):
#             zodiac_ranges = {
#                 1: ((3, 21), (4, 19)),      # Aries
#                 2: ((4, 20), (5, 20)),      # Taurus
#                 3: ((5, 21), (6, 21)),      # Gemini
#                 4: ((6, 22), (7, 22)),      # Cancer
#                 5: ((7, 23), (8, 22)),      # Leo
#                 6: ((8, 23), (9, 22)),      # Virgo
#                 7: ((9, 23), (10, 23)),     # Libra
#                 8: ((10, 24), (11, 22)),    # Scorpio
#                 9: ((11, 23), (12, 21)),    # Sagittarius
#                 10: ((12, 22), (1, 19)),    # Capricorn
#                 11: ((1, 20), (2, 18)),     # Aquarius
#                 12: ((2, 19), (3, 20))      # Pisces
#             }
#             for zodiac, ((start_month, start_day), (end_month, end_day)) in zodiac_ranges.items():
#                 if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
#                     return zodiac
#             return None
        
#         player['month'] = pd.to_datetime(player['birthday']).dt.month
#         player['day'] = pd.to_datetime(player['birthday']).dt.day
#         player['zodiac_label'] = player.apply(lambda row: get_zodiac(row['month'], row['day']), axis=1)
#         truth = player['zodiac_label'].astype(int).tolist()
#         player = player.drop(['day', 'month', 'zodiac_label'], axis=1)

#         start_time = time.time()
#         op = LogicalGroupBy(operand_type=OperandType.ROW)
#         pred = op.execute(impl_type=args.impl,
#                             condition="Group the birthdate by zodiacs",
#                             df=player,
#                             depend_on=['birthday'],
#                             thinking=args.thinking)
#         pred = pred['cluster_label'].astype(int).tolist()
#         end_time = time.time()

#         ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
#         return (ari_score, nmi_score), end_time-start_time, op.get_tokens()
    
#     query = syn_groupby_eval
#     operator = OperatorType.GROUPBY

#     for impl in [ImplType.LLM_ONLY, ImplType.LLM_SEMI]:
#         for thinking in [False, True]:
#             args = argparse.Namespace(operator=operator, impl=impl, thinking=thinking,  
#                                       start = 'syn0', end = 'syn0')
#             file_name = args_to_filename(args, ".csv")
#             vary_size_exp(operator, file_name, query, ns, args)


# def order_scale():
#     def syn_order_eval(args):
#         query = "Sort their ages in ascending order."
#         player = pd.read_csv("./databases/european_football_2/Player.csv")
#         player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#         player = player[:args.n]
        
#         truth = player.sort_values(by = 'birthday', ascending=False)['player_name'].tolist()

#         start_time = time.time()
#         op = LogicalOrder(operand_type=OperandType.ROW)
#         pred = op.execute(impl_type=args.impl,
#                             condition="the age of the player",
#                             depend_on= 'birthday',
#                             ascending = True,
#                             k = len(player),
#                             df = player, 
#                             sort_algo = args.sort_algo,
#                             thinking = args.thinking)
#         pred = pred['player_name'].tolist()
#         end_time = time.time()

#         acc, _, _ = f1_score(truth, pred)
#         tau = kendall_tau_at_k(truth, pred)
#         return (acc, tau), end_time-start_time, op.get_tokens()
    
#     query = syn_order_eval
#     operator = OperatorType.ORDER
    
#     for impl in [ImplType.LLM_ONLY, ImplType.LLM_SEMI, ImplType.LLM_SEMI_OPTIM]:
#         for thinking in [False, True]:
#             args = argparse.Namespace(operator=operator, impl=impl, thinking=thinking,  
#                                       start = 'syn0', end = 'syn0', sort_algo=None)
#             if impl == ImplType.LLM_SEMI:
#                 for sort_algo in ['simple', 'heap']:
#                     setattr(args, 'sort_algo', sort_algo)
#                     file_name = args_to_filename(args, ".csv")
#                     vary_size_exp(operator, file_name, query, ns, args)
#                 continue

#             file_name = args_to_filename(args, ".csv")
#             vary_size_exp(operator, file_name, query, ns, args)

# def map_scale():
#     def syn_map_eval(args):
#         query = "Map their birthdates to the zodiacs."
#         player = pd.read_csv("./databases/european_football_2/Player.csv")
#         player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#         player = player[:args.n]

#         def get_zodiac(month, day):
#             zodiac_ranges = {
#                 'Aries': ((3, 21), (4, 19)),    
#                 'Taurus': ((4, 20), (5, 20)),      
#                 'Gemini': ((5, 21), (6, 21)),     
#                 'Cancer': ((6, 22), (7, 22)),    
#                 'Leo': ((7, 23), (8, 22)),      
#                 'Virgo': ((8, 23), (9, 22)),      
#                 'Libra': ((9, 23), (10, 23)),     
#                 'Scorpio': ((10, 24), (11, 22)),    
#                 'Sagittarius': ((11, 23), (12, 21)),   
#                 'Capricorn': ((12, 22), (1, 19)),    
#                 'Aquarius': ((1, 20), (2, 18)),    
#                 'Pisces': ((2, 19), (3, 20))      
#             }
#             for zodiac, ((start_month, start_day), (end_month, end_day)) in zodiac_ranges.items():
#                 if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
#                     return zodiac
#             return None
        
#         player['month'] = pd.to_datetime(player['birthday']).dt.month
#         player['day'] = pd.to_datetime(player['birthday']).dt.day
#         player['zodiac'] = player.apply(lambda row: get_zodiac(row['month'], row['day']), axis=1)
#         truth = player[['player_name', 'zodiac']]
#         player = player.drop(['day', 'month', 'zodiac'], axis=1)

#         right_table = pd.DataFrame({
#             "zodiac": [
#                 "Aries", "Taurus", "Gemini", "Cancer", "Leo", 
#                 "Virgo", "Libra", "Scorpio", "Sagittarius", 
#                 "Capricorn", "Aquarius", "Pisces"
#         ]})

#         start_time = time.time()
#         op = LogicalMap(operand_type=OperandType.CELL)
#         pred = op.execute(impl_type=args.impl,
#             condition="The 'birthday' matches the 'zodiac'", 
#             left_df=player, 
#             right_df=right_table, 
#             left_on='birthday', 
#             right_on='zodiac',
#             thinking = args.thinking)
#         pred = pred[['left_player_name', 'right_zodiac']]
#         end_time = time.time()

#         return f1_score(truth, pred), end_time-start_time, op.get_tokens()

#     query = syn_map_eval
#     operator = OperatorType.MAP
    
#     for impl in [ImplType.LLM_SEMI]:
#         for thinking in [False, True]:
#             args = argparse.Namespace(operator=operator, impl=impl, thinking=thinking,  
#                                       start = 'syn0', end = 'syn0', sort_algo=None)

#             file_name = args_to_filename(args, ".csv")
#             vary_size_exp(operator, file_name, query, ns, args)

# def token_cal():
#     import json
#     player = pd.read_csv("./databases/european_football_2/Player.csv")
#     player = player.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#     player = player[['birthday']]
#     rows = [json.dumps(row) for row in player.reset_index().to_dict(orient='records')]

#     # print(len(list(rows)))

#     enc = tiktoken.encoding_for_model("gpt-4o")
#     curr_token = 0
#     tokens = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

#     j = 0
#     for i, row in enumerate(rows):
#         curr_token += len(enc.encode(str(row)))
#         if curr_token > tokens[j]:
#             print(f"The first {i} rows: {curr_token}")
#             j += 1
#         if j >= len(tokens):
#             break


if __name__ == '__main__':
    random_seeds = [1342, 27, 336, 19881, 584, 249, 1037, 98, 1950, 63]
    for seed in random_seeds:
        # select_scale(seed)
        impute_scale(seed)
