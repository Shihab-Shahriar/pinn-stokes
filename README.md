## Organization

data/ (part of git repo)
    - dataset files 
    - weight files

src/
    - All MFS base code
    - Two body and 3-body dataset generation code (all shapes)
    - Implementation of different flavors of RPY

experiments/
    - All training notebooks
    - Accuracy/ analysis 
    - ablation study notebooks (SPSD, PINN vs NN)

benchmarks/
    - Targeted benchmark: RPY vs NN vs MFS for single application of mobility operator using a O(N^2) algorithm.
    - Full simulation of a real-world problem, with FMM. Demonstrate accuracy advantage of NN vs RPY. Show performance penalty. 

utils/
    - plotting, timing etc