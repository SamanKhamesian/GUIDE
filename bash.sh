#!/bin/bash

# Define the array of patient IDs
PATIENT_IDS=(1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
#PATIENT_ID_LIST=(540 544 552 559 563 567 570 575 584 588 591 596)

SEEDS=(42 43 44 45 46)

for patient_id in "${PATIENT_IDS[@]}"
do
  for seed in "${SEEDS[@]}"
  do
    python3 -m td3_bc_model.main "$patient_id" "$seed"
  done
done