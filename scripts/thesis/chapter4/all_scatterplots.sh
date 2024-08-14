models=('model3' 'model10' 'model2' 'Wang')

for model in ${models[@]};
do
    python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0a/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots_log_a/${model}_a --log_a   &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0b/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots_log_a/${model}_b --log_a &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0b/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots_log_a/${model}_c --adjust_kinetics --log_a --subtraction_df thesis_data/25112022_MW_FF_processed/subtraction_qc.csv --reversal -89.83 &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0d/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots_log_a/${model}_d --log_a &&\
            python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/CaseII/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots_log_a/${model}_II --log_a
    done;

for model in ${models[@]};
do
    python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0a/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots/${model}_a  &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0b/${model}/combine_fitting_results/combined_fitting_results.csv  thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots/${model}_b  &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0b/${model}/combine_fitting_results/combined_fitting_results.csv  thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/scatterplots/${model}_c --adjust_kinetics --subtraction_df thesis_data/25112022_MW_FF_processed/subtraction_qc.csv --reversal -89.83 &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/Case0d/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/${model}_d &&\
        python3 scripts/thesis/chapter4/scatterplots.py thesis_data/sydney_fitting/25112022MW/CaseII/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/chrono.txt --model ${model} --ignore_protocols longap -o thesis_plots/chapter4/${model}_II
done;
