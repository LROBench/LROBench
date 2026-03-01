#!/bin/bash

python main.py --operator order --impl llm_all --end 30 --out_dir "./result/order/${MODEL}/"
python main.py --operator order --impl llm_one --end 30 --sort_algo simple --out_dir "./result/order/${MODEL}/"
python main.py --operator order --impl llm_one --end 30 --sort_algo heap --out_dir "./result/order/${MODEL}/"
python main.py --operator order --impl llm_semi --end 30 --out_dir "./result/order/${MODEL}/" 

python main.py --operator order --impl llm_all --end 30 --out_dir "./result/order/${MODEL}/" --thinking
python main.py --operator order --impl llm_one --end 30 --sort_algo simple --out_dir "./result/order/${MODEL}/" --thinking
python main.py --operator order --impl llm_one --end 30 --sort_algo heap --out_dir "./result/order/${MODEL}/" --thinking
python main.py --operator order --impl llm_semi --end 30 --out_dir "./result/order/${MODEL}/" --thinking

