#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <time.h>

#define GRID_SIZE 30
#define GENERATIONS 100
#define ALIVE '*'
#define DEAD  ' '

void initialize_grid(int *grid, int rows, int cols, int rank) {
    unsigned int seed = time(NULL) + rank;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            grid[i * cols + j] = rand_r(&seed) % 4 == 0;  // ~25% alive
}

int count_neighbors(int *grid, int rows, int cols, int row, int col) {
    int count = 0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int r = row + i;
            int c = col + j;
            if (r >= 0 && r < rows && c >= 0 && c < cols)
                count += grid[r * cols + c];
        }
    return count;
}

void update_grid(int *current, int *next, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j;
            int neighbors = count_neighbors(current, rows, cols, i, j);
            next[index] = (current[index] && (neighbors == 2 || neighbors == 3)) || 
              (!current[index] && neighbors == 3);
        }
    }
}

void print_grid(int *local_grid, int local_rows, int rank, int size) {
    int *global_grid = NULL;
    if (rank == 0) {
        global_grid = malloc(GRID_SIZE * GRID_SIZE * sizeof(int));
    }

    MPI_Gather(local_grid + GRID_SIZE, local_rows * GRID_SIZE, MPI_INT,
               global_grid, local_rows * GRID_SIZE, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\033[H\033[2J");  // Clear screen and move cursor to top
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++)
                putchar(global_grid[i * GRID_SIZE + j] ? ALIVE : DEAD);
            putchar('\n');
        }
        fflush(stdout);
        usleep(100000);  // 100 ms delay for animation
        free(global_grid);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = GRID_SIZE / size;
    int remainder = GRID_SIZE % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    local_rows += 2;  // Add ghost rows

    int *current = malloc(local_rows * GRID_SIZE * sizeof(int));
    int *next = malloc(local_rows * GRID_SIZE * sizeof(int));

    initialize_grid(current + GRID_SIZE, local_rows - 2, GRID_SIZE, rank);

    if (rank == 0) setbuf(stdout, NULL);  // Disable buffering for real-time display

    for (int gen = 0; gen < GENERATIONS; gen++) {
        // Exchange ghost rows
        if (rank > 0) {
            MPI_Sendrecv(current + GRID_SIZE, GRID_SIZE, MPI_INT, rank - 1, 0,
                         current, GRID_SIZE, MPI_INT, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(current + (local_rows - 2) * GRID_SIZE, GRID_SIZE, MPI_INT, rank + 1, 0,
                         current + (local_rows - 1) * GRID_SIZE, GRID_SIZE, MPI_INT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        update_grid(current, next, local_rows, GRID_SIZE);

        if (gen % 5 == 0) {
            print_grid(current, local_rows - 2, rank, size);
        }

        int *temp = current;
        current = next;
        next = temp;
    }

    if (rank == 0) {
        printf("\nSimulation completed. Press Enter to exit...\n");
        getchar();  // Pause before closing
    }

    free(current);
    free(next);
    MPI_Finalize();
    return 0;
}
