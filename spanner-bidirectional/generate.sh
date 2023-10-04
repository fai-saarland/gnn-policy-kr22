#!/bin/bash

for SPANNERS in 1 3 5 7 9
do
  for LOCATIONS in 1 5 10 15 20 25 30 35 40 100 200 300
  do
    echo "${SPANNERS} ${LOCATIONS}"
    python3 spanner-generator.py ${SPANNERS} ${SPANNERS} ${LOCATIONS} > "prob_spanners-${SPANNERS}_nuts-${SPANNERS}_locations-${LOCATIONS}.pddl"
  done
done
