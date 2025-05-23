#!/bin/bash
echo "🧬 Game of Life Performance Analysis"
echo "=================================="

configurations=(
    "1 8"   # 1 MPI process, 8 OMP threads
    "2 4"   # 2 MPI processes, 4 OMP threads each
    "4 2"   # 4 MPI processes, 2 OMP threads each
    "8 1"   # 8 MPI processes, 1 OMP thread each
)

for config in "${configurations[@]}"; do
    read -r np nt <<< "$config"
    echo "Testing: $np MPI processes × $nt OpenMP threads"
    time OMP_NUM_THREADS=$nt mpirun -np $np ./game_of_life
    echo "---"
done
