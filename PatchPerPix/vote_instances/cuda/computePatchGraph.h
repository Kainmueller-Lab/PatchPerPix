#ifndef COMPUTEPATCHGRAPH_H
#define COMPUTEPATCHGRAPH_H

#include <cstdint>

__global__ void computePatchGraph(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const float inCons[][PSY*2][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float affGraph[], const unsigned pairsIDs[], const uint64_t numPairs);

#endif /* COMPUTEPATCHGRAPH_H */
