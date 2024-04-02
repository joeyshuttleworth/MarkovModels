models=('model3' 'model10' 'model2' 'Wang')

for model in ${models};
do
	python3 scripts/scatterplots.py data/sydney_fitting/25112022MW/Case0a/${model}/combine_fitting_results/combined_fitting_results.csv --model {$model} --ignore_protocols longap -o output/scatterplots/${model}_a &&\
	python3 scripts/scatterplots.py data/sydney_fitting/25112022MW/Case0b/${model}/combine_fitting_results/combined_fitting_results.csv --model {$model} --ignore_protocols longap -o output/scatterplots/${model}_b &&\
	python3 scripts/scatterplots.py data/sydney_fitting/25112022MW/Case0c/${model}/combine_fitting_results/combined_fitting_results.csv --model {$model} --ignore_protocols longap -o output/scatterplots/${model}_c --adjust_kinetics
done;



