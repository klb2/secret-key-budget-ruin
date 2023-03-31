#!/bin/sh

# Run all script to generate the results presented in the paper "Reliability
# and Latency Analysis for Wireless Communication Systems with a Secret-Key
# Budget" (Karl-L. Besser, Rafael Schaefer, and Vincent Poor, Apr 2023).
#
# Copyright (C) 2023 Karl-Ludwig Besser
# License: GPLv3

echo "Figure 1: Illustration of temporal progress"
python3 illustration_budget.py --plot

echo "Figure 3: Outage probability for different initial budgets"
python3 simulation_deterministic.py -vv -b 5 10 20 50 -t 3 4 5 10 --plot --export

echo "Figure 6: Comparison of outage probability in finite time and ultimate ruin"
python3 simulation_random_timing.py -v -b 20 -m 30 -p .1 --plot --export
python3 simulation_random_timing.py -v -b 20 -m 30 -p .35 --plot --export
