#ifndef RANKPATCHES_H
#define RANKPATCHES_H

__global__ void rankPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const float inCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outScore[][DATAYSIZE][DATAXSIZE]);

#endif /* RANKPATCHES_H */
