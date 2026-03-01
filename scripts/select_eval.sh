#!/bin/bash

implementations=("llm_all" "llm_one")

for impl in "${implementations[@]}"; do
    python main.py --operator select --impl $impl --start 100 --end 130 --out_dir "./result/select/${MODEL}/"
    python main.py --operator select --impl $impl --start 100 --end 130 --out_dir "./result/select/${MODEL}/" --thinking 

    for example_num in "0" "3"; do
        python main.py --operator select --impl $impl --start 200 --end 215 --out_dir "./result/select/${MODEL}/" --example_num $example_num
        python main.py --operator select --impl $impl --start 200 --end 215 --out_dir "./result/select/${MODEL}/" --example_num $example_num --thinking

        python main.py --operator select --impl $impl --start 300 --end 315 --out_dir "./result/select/${MODEL}/" --example_num $example_num
        python main.py --operator select --impl $impl --start 300 --end 315 --out_dir "./result/select/${MODEL}/" --example_num $example_num --thinking
    done
done
