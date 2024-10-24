#!/bin/bash

datasets=("adult" "arrhythmia" "compas" "crime" "drug_consumption" "drug_consumption_multi" "german_credit" "hrs" "lsac" "obesity" "parkinsons_updrs" "student_performance")
for i in {0..11}
do
  python experiment.py --dataset ${datasets[i]}
done
