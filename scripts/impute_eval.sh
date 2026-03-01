#!/bin/bash

python main.py --operator impute --impl llm_all --start 100 --end 130 --out_dir "./result/impute/${MODEL}/"
python main.py --operator impute --impl llm_all --start 100 --end 130 --out_dir "./result/impute/${MODEL}/" --thinking
python main.py --operator impute --impl llm_one --start 100 --end 130 --out_dir "./result/impute/${MODEL}/" --example_num 0
python main.py --operator impute --impl llm_one --start 100 --end 130 --out_dir "./result/impute/${MODEL}/" --example_num 0 --thinking
python main.py --operator impute --impl llm_one --start 100 --end 130 --out_dir "./result/impute/${MODEL}/" --example_num 3
python main.py --operator impute --impl llm_one --start 100 --end 130 --out_dir "./result/impute/${MODEL}/" --example_num 3 --thinking


python main.py --operator impute --impl llm_all --start 200 --end 230 --out_dir "./result/impute/${MODEL}/"
python main.py --operator impute --impl llm_all --start 200 --end 230 --out_dir "./result/impute/${MODEL}/" --thinking
python main.py --operator impute --impl llm_one --start 200 --end 230 --out_dir "./result/impute/${MODEL}/"
python main.py --operator impute --impl llm_one --start 200 --end 230 --out_dir "./result/impute/${MODEL}/" --thinking


python main.py --operator impute --impl llm_all --start 300 --end 315 --out_dir "./result/impute/${MODEL}/" --example_num 0
python main.py --operator impute --impl llm_all --start 300 --end 315 --out_dir "./result/impute/${MODEL}/" --example_num 3
python main.py --operator impute --impl llm_all --start 300 --end 315 --out_dir "./result/impute/${MODEL}/" --example_num 0 --thinking
python main.py --operator impute --impl llm_all --start 300 --end 315 --out_dir "./result/impute/${MODEL}/" --example_num 3 --thinking