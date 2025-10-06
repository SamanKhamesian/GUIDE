#!/bin/bash

# Define the array of patient IDs
PATIENT_ID_LIST=(1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
#PATIENT_ID_LIST=(540 544 552 559 563 567 570 575 584 588 591 596)

# Loop through the array
for i in "${PATIENT_ID_LIST[@]}"
do
  # Run the Python script and pass the current patient ID as an argument
    python3 -m td3_bc_model.main "$i"
done