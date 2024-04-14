#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static inline int64_t max(int x, int y) {
    return x > y ? x : y;
}

int64_t evaluate(int64_t *sol, int64_t **mj_table, int64_t num_machines, int64_t num_jobs) {
    int64_t *machine_cache = (int64_t*)calloc(num_machines, sizeof(int64_t));

    for (int i = 0; i < num_jobs; i++) {
        for (int machine_id = 0; machine_id < num_machines; machine_id++) {
            if (machine_id != 0) {
                machine_cache[machine_id] = max(machine_cache[machine_id - 1], machine_cache[machine_id]) + mj_table[machine_id][sol[i]];
            } else {
                machine_cache[machine_id] = machine_cache[machine_id] + mj_table[machine_id][sol[i]];
            }
        }
    }

    int64_t makespan = machine_cache[num_machines - 1];
    free(machine_cache);
    return makespan;
}