models=('model3' 'model10' 'model2' 'Wang')
cases=('CaseII', 'Case0d')

for model in ${models[@]};
do
    for case in ${cases[@]};
        do
            echo ${case} ${model}
            python3 scripts/thesis/chapter4/optimisation_results.py thesis_data/25112022_MW_FF_processed/traces ${case} thesis_data/sydney_fitting/25112022MW/${case}/${model}/combine_fitting_results/combined>
        done;
    done;
