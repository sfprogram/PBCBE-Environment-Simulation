 # Frog vs Snake Simulation
 
Frogs attempt to avoid snakes and walls while collecting flies 
and eventually going into water to reproduce.

## [Report](Simulation_Report.pdf)

Please click the link to read the report for a detailed explanation of this project
or find it in the repo ```Simulation_Report.pdf```


## Algorithms

### Genetic Algorithm (GA)
Three selection methods have been implemented:
- Tournament
- Steady state
- Simulated annealing with exponential cooling

### Population-Based Consensus Behavioural Evolution (PBCBE)
Noval take on consensus evolutionary algorithms. Uses the same gene representation but will determine how to evolve each gene independently
based on the recorded behaviour during collected during a generation.


## Use

### Main 
This is the standalone implementation of both GA and PBCBE which shows the visuals and has
the learning and evolution logic.


### Training
Headless version of main for faster training without the visual overhead. Results are saved
to ```training data/GA or Consensus/```


### Generation_view
Purely visual basin to load ```training data``` into tkinter. Currently is set to the latest generation
in the dataset.


### Plot
Plots the data stored within ```training data``` respectively. Saves the plotted graphs to ```plots/GA 
or Consensus/```


## Installation and setup
1. Clone repository
2. Install dependencies
3. Run either main or training to simulate
4. Optional to plot and/or use generation_view

