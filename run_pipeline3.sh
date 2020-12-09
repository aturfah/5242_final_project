#!/bin/bash
for temp in {1..480..1}
do
    clear
    python train_models3.py
done

python parse_results.py

[ -d images ] || mkdir images

./generate_base_model_output.R > base_tables.txt 
./generate_other_model_output.R > other_tables.txt 

