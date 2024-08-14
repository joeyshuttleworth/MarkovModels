models=('model3' 'model10' 'model2' 'Wang')
cases=('Case0b' 'Case0a' 'Case0d' 'CaseII')

for model in ${models[@]};
do
for case in ${cases[@]};
do
    echo ${case} ${model}
	python3 scripts/thesis/chapter4/optimisation_results.py thesis_data/25112022_MW_FF_processed/traces ${case} thesis_data/sydney_fitting/25112022MW/${case}/${model}/combine_fitting_results/combined_fitting_results.csv thesis_data/25112022_MW_FF_processed/subtraction_qc.csv ${model} --output thesis_plots/chapter4/optimisation/${model}_${case} -c 20 --experiment_name 25112022_MW  || exit 1
done;
done;


