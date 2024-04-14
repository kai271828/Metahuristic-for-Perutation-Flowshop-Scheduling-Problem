#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static inline int32_t max(int x, int y) {
    return x > y ? x : y;
}

int32_t evaluate(int32_t *sol, int32_t **mj_table, int32_t num_machines, int32_t num_jobsf) {
    int32_t *machine_cache = (int32_t*)calloc(num_machines, sizeof(int32_t));

    for (int i = 0; i < num_jobs; i++) {
        for (int machine_id = 0; machine_id < num_machines; machine_id++) {
            if (machine_id != 0) {
                machine_cache[machine_id] = max(machine_cache[machine_id - 1], machine_cache[machine_id]) + mj_table[machine_id][sol[i]];
            } else {
                machine_cache[machine_id] = machine_cache[machine_id] + mj_table[machine_id][sol[i]];
            }
        }
    }

    int32_t makespan = machine_cache[num_machines - 1];
    free(machine_cache);
    return makespan;
}