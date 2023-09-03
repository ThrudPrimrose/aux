#include <hwloc.h>
#include <stdio.h>

int main(void) {
    hwloc_topology_t topology;
    int nbcores, nbpci;
    int depth;
    unsigned i, n;
    unsigned long size;
    int levels;
    char string[128];
    int topodepth;
    void *m;
    hwloc_topology_init(&topology);  // initialization
    hwloc_topology_load(topology);   // actual detection
    topodepth = hwloc_topology_get_depth(topology);

    for (depth = 0; depth < topodepth; depth++) {
        printf("*** Objects at level %d\n", depth);
        for (i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth);
                i++) {
            hwloc_obj_type_snprintf(string, sizeof(string),
                    hwloc_get_obj_by_depth(topology, depth, i), 0);
            printf("Index %u: %s\n", i, string);
        }
    }

    nbcores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    printf("%d cores\n", nbcores);

    nbpci =  hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_OS_DEVICE);
    printf("%d nbpci\n", nbpci);
    
    hwloc_topology_destroy(topology);

    return 0;
}