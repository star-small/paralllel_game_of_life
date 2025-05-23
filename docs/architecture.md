# Architecture Documentation - Hybrid Game of Life

## System Overview

This document describes the architectural design of the hybrid MPI+OpenMP implementation of Conway's Game of Life, detailing the parallel decomposition strategy, communication patterns, and synchronization mechanisms.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Game of Life Grid                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │   │    |
│  │ │Process 0│ │Process 1│ │Process 2│ │Process 3│ │   │    |
│  │ │ (Rank)  │ │ (Rank)  │ │ (Rank)  │ │ (Rank)  │ │   │    |
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │   │    |
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │   │    |
│  │ │ Thread  │ │ Thread  │ │ Thread  │ │ Thread  │ │   │    |
│  │ │ Pool    │ │ Pool    │ │ Pool    │ │ Pool    │ │   │    |
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │   │    |
│  └─────────────────────────────────────────────────────┘    |
└─────────────────────────────────────────────────────────────┘
```

## Decomposition Strategy

### 1. Domain Decomposition (MPI Level)

#### Horizontal Partitioning
The 2D grid is divided into horizontal strips, with each MPI process responsible for a contiguous set of rows.

```
Original Grid (6×6):        Process Assignment:
┌─────────────────────┐     ┌─────────────────────┐
│ 0  1  2  3  4  5   │     │ 0  0  0  0  0  0     │ ← Process 0
│ 6  7  8  9 10 11   │     │ 0  0  0  0  0  0     │
│12 13 14 15 16 17   │     │ 1  1  1  1  1  1     │ ← Process 1
│18 19 20 21 22 23   │     │ 1  1  1  1  1  1     │
│24 25 26 27 28 29   │     │ 2  2  2  2  2  2     │ ← Process 2
│30 31 32 33 34 35   │     │ 2  2  2  2  2  2     │
└─────────────────────┘     └─────────────────────┘
```

#### Load Balancing Algorithm
```c
int rows_per_proc = GRID_SIZE / size;
int remainder = GRID_SIZE % size;
int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
```

This ensures even distribution when grid size doesn't divide evenly among processes.

### 2. Ghost Cell Management

#### Ghost Row Structure
Each process maintains ghost rows from neighboring processes to handle boundary calculations.

```
Process 1 Memory Layout:
┌─────────────────────┐
│   Ghost Row (Top)   │ ← From Process 0
├─────────────────────┤
│   Local Data Row 1  │
│   Local Data Row 2  │
│       ...           │
│   Local Data Row N  │
├─────────────────────┤
│ Ghost Row (Bottom)  │ ← From Process 2
└─────────────────────┘
```

#### Memory Allocation
```c
// Total rows = local_rows + 2 ghost rows
int total_rows = local_rows + 2;
int *grid = malloc(total_rows * GRID_SIZE * sizeof(int));

// Local data starts at offset GRID_SIZE (skip top ghost row)
int *local_data = grid + GRID_SIZE;
```

## Thread-Level Parallelization (OpenMP)

### 1. Parallel Regions

#### Grid Initialization
```c
#pragma omp parallel for collapse(2)
for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
        grid[(i+1) * GRID_SIZE + j] = rand_r(&seed) % 4 == 0;
    }
}
```

#### Grid Update Computation
```c
#pragma omp parallel for collapse(2)
for (int i = 1; i < rows - 1; i++) {  // Skip ghost rows
    for (int j = 0; j < GRID_SIZE; j++) {
        int index = i * GRID_SIZE + j;
        int neighbors = count_neighbors(current, rows, GRID_SIZE, i, j);
        next[index] = apply_game_rules(current[index], neighbors);
    }
}
```

### 2. Thread-Safe Operations

#### Random Number Generation
Each thread uses a separate seed to avoid race conditions:
```c
unsigned int seed = time(NULL) + rank + omp_get_thread_num();
grid[index] = rand_r(&seed) % 4 == 0;
```

#### Memory Access Patterns
- **No shared writes**: Each thread writes to distinct memory locations
- **Read-only sharing**: Ghost rows are read-only during computation
- **Cache-friendly**: Consecutive memory access within each thread

## Communication Patterns

### 1. Boundary Exchange Protocol

#### Synchronous Communication
```c
// Exchange with upper neighbor
if (rank > 0) {
    MPI_Sendrecv(
        current + GRID_SIZE,                    // Send first local row
        GRID_SIZE, MPI_INT, rank - 1, 0,
        current,                                // Receive into top ghost row
        GRID_SIZE, MPI_INT, rank - 1, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// Exchange with lower neighbor
if (rank < size - 1) {
    MPI_Sendrecv(
        current + (local_rows) * GRID_SIZE,     // Send last local row
        GRID_SIZE, MPI_INT, rank + 1, 0,
        current + (local_rows + 1) * GRID_SIZE, // Receive into bottom ghost row
        GRID_SIZE, MPI_INT, rank + 1, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}
```

### 2. Communication Topology

```
Process 0 ←→ Process 1 ←→ Process 2 ←→ Process 3
    ↑                                      ↓
    └──── Collective Operations ──────────┘
```

#### Point-to-Point Communication
- **Purpose**: Ghost row exchange
- **Pattern**: Nearest neighbor
- **Frequency**: Every generation
- **Volume**: 2 × GRID_SIZE integers per process per generation

#### Collective Communication
- **Purpose**: Global grid visualization
- **Pattern**: Gather operation
- **Frequency**: Every N generations (for display)
- **Volume**: Full grid size

## Synchronization Mechanisms

### 1. MPI Synchronization

#### Implicit Synchronization
- `MPI_Sendrecv` provides implicit synchronization between neighbors
- `MPI_Gather` synchronizes all processes for visualization

#### Explicit Barriers
```c
// Not used in main computation loop to avoid unnecessary overhead
// MPI_Barrier(MPI_COMM_WORLD);  // Only for debugging
```

### 2. OpenMP Synchronization

#### Implicit Barriers
- End of parallel regions
- Work-sharing constructs (`#pragma omp for`)

#### Thread-Safe Memory Management
```c
// Thread-local arrays for computations
#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    // Each thread processes independent data chunks
}
```

## Memory Management

### 1. Data Structures

#### Grid Representation
```c
// 1D array with 2D indexing for cache efficiency
int *grid = malloc((local_rows + 2) * GRID_SIZE * sizeof(int));

// Access pattern: grid[row * GRID_SIZE + col]
#define GRID_INDEX(row, col) ((row) * GRID_SIZE + (col))
```

#### Double Buffering
```c
int *current_grid = malloc(total_size);
int *next_grid = malloc(total_size);

// Swap pointers instead of copying data
int *temp = current_grid;
current_grid = next_grid;
next_grid = temp;
```

### 2. Memory Layout Optimization

#### NUMA Considerations
```c
// First-touch policy: data allocated where first accessed
#pragma omp parallel for
for (int i = 0; i < local_rows; i++) {
    // This thread will likely access this memory later
    initialize_row(grid, i);
}
```

#### Cache-Line Alignment
```c
// Align allocations to cache line boundaries (64 bytes)
int *aligned_grid = aligned_alloc(64, total_size * sizeof(int));
```

## Error Handling and Robustness

### 1. MPI Error Handling
```c
int rc = MPI_Sendrecv(/* parameters */);
if (rc != MPI_SUCCESS) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    MPI_Error_string(rc, error_string, &length);
    fprintf(stderr, "MPI Error: %s\n", error_string);
    MPI_Abort(MPI_COMM_WORLD, rc);
}
```

### 2. Resource Management
```c
// Cleanup on exit
void cleanup() {
    free(current_grid);
    free(next_grid);
    MPI_Finalize();
}

// Register cleanup function
atexit(cleanup);
```

## Performance Considerations

### 1. Communication Optimization

#### Overlapping Communication and Computation
```c
// Future optimization: use non-blocking communication
MPI_Isend(boundary_data, ...);
// Compute interior cells while communication proceeds
compute_interior_cells();
MPI_Wait(&request, &status);
// Compute boundary cells after communication completes
compute_boundary_cells();
```

#### Communication Volume Reduction
- **Current**: Send full boundary rows
- **Optimization**: Could pack only necessary boundary data

### 2. Computational Optimization

#### Loop Optimization
```c
// Cache-friendly access pattern
for (int i = 1; i < rows - 1; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
        // Access grid[i][j] and its neighbors
        // All accesses are spatially local
    }
}
```

#### Branch Prediction
```c
// Game of Life rules optimized for branch prediction
next[index] = (current[index] && (neighbors == 2 || neighbors == 3)) ||
              (!current[index] && neighbors == 3);
```

## Scalability Analysis

### 1. Complexity Analysis

#### Computational Complexity
- **Per iteration**: O(N²) where N is grid dimension
- **Total**: O(G × N²) where G is number of generations

#### Communication Complexity
- **Per iteration**: O(N) for boundary exchange
- **Total**: O(G × N) for boundary exchange + O(G × N²) for visualization

### 2. Scaling Limits

#### Strong Scaling Limits
- **Communication overhead**: Increases as O(P) where P is process count
- **Amdahl's Law**: Serial fraction limits maximum speedup
- **Load imbalance**: Becomes significant with many processes

#### Weak Scaling Limits
- **Communication-to-computation ratio**: Remains constant with proper scaling
- **Memory bandwidth**: May become bottleneck on shared memory systems

## Configuration Guidelines

### 1. Process-Thread Mapping

#### Recommended Configurations
```bash
# Small systems (≤8 cores)
OMP_NUM_THREADS=8 mpirun -np 1 ./game_of_life

# Medium systems (8-32 cores)
OMP_NUM_THREADS=4 mpirun -np 8 ./game_of_life

# Large systems (>32 cores)
OMP_NUM_THREADS=2 mpirun -np 16 ./game_of_life
```

#### Hardware-Specific Tuning
```bash
# NUMA-aware placement
export OMP_PROC_BIND=true
export OMP_PLACES=cores
numactl --cpunodebind=0 mpirun -np 4 ./game_of_life
```

### 2. Problem Size Scaling

#### Grid Size Recommendations
- **Minimum**: 100×100 per process for meaningful work
- **Optimal**: 1000×1000+ for strong scaling studies
- **Maximum**: Limited by available memory

