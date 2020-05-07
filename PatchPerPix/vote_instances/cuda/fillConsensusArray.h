#ifndef FILLCONSENSUSARRAY_H
#define FILLCONSENSUSARRAY_H

#include <cstdint>

__global__ void fillConsensusArray_allPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    //const bool inFg[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE]);

__global__ void fillConsensusArray_subsetPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    //const bool inFg[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const unsigned patchesIDs[], const uint64_t numPatches);

#endif /* FILLCONSENSUSARRAY_H */
