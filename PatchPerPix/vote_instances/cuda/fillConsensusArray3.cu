#include <cstdint>

// sets of fg/bg pixels in python not sorted, so slightly different result
// here, total sum over array should be identical
__device__ void _fillConsensusArray3(
    unsigned idx, unsigned idy, unsigned idz,
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
{
    unsigned int mid = int((PSX*PSY*PSZ)/2);
    unsigned const PSXH = int(PSX/2);
    unsigned const PSYH = int(PSY/2);
    unsigned const PSZH = int(PSZ/2);
    // ignore border pixels
    if ((idx < (DATAXSIZE-PSXH)) &&
        (idy < (DATAYSIZE-PSYH)) &&
        (idz < (DATAZSIZE-PSZH)) &&
        (idx >= (PSXH)) &&
        (idy >= (PSYH)) &&
        (idz >= (PSZH))){
        // only if pixel in foreground
        if(inPred[mid][idz][idy][idx] <= TH)
            return;

        // for all pairs of pixels in patch
        for(int pz1 = 0; pz1 < PSZ; pz1++) {
            for(int py1 = 0; py1 < PSY; py1++) {
                for(int px1 = 0; px1 < PSX; px1++) {
                    // offset in patch pixel 1
                    int po1 = px1 + PSX * py1 + PSX * PSY * pz1;

                    // first element of pair should have high affinity
                    // (to not count every pair twice)
                    float v1 = inPred[po1][idz][idy][idx];
                    if(v1 <= TH) {
                        continue;
                    }
                    // check if predicted affinity in patch agrees
                    // with corresponding pixel in fg prediction
                    const int z1 = idz+pz1-PSZH;
                    const int y1 = idy+py1-PSYH;
                    const int x1 = idx+px1-PSXH;
                    if(inPred[mid][z1][y1][x1] <= TH) {
                        continue;
                    }

                    if(inOverlap[z1][y1][x1] != 0){
                        continue;
                    }


                    // second element of pixel pair
                    for(int pz2 = 0; pz2 < PSZ; pz2++) {
                        for(int py2 = 0; py2 < PSY; py2++) {
                            for(int px2 = 0; px2 < PSX; px2++) {
                                // offset in patch pixel 2
                                int po2 = px2 + PSX * py2 + PSX * PSY * pz2;

                                if (po1 == po2)
                                    continue;

                                const int z2 = idz+pz2-PSZH;
                                const int y2 = idy+py2-PSYH;
                                const int x2 = idx+px2-PSXH;
                                // patch pixel should correspond to foreground
                                if(inPred[mid][z2][y2][x2] <= TH) {
                                    continue;
                                }

                                if(inOverlap[z2][y2][x2] != 0){
                                    continue;
                                }

                                float v2 = inPred[po2][idz][idy][idx];
                                // offset from pixel 1 to pixel 2
                                int zo = pz2-pz1+PSZ-1;
                                int yo = py2-py1+PSY-1;
                                int xo = px2-px1+PSX-1;

                                // if both high affinity, increase consensus
                                // pixel 1 with offset yo/xo to pixel 2
                                if(v2 > TH) {
                                    if(po2 <= po1)
                                        continue;
                                    // atomicAdd(
                                    //     &outCons[zo][yo][xo][z1][y1][x1],
                                    //     1);
                                    // float v3 = v1*v2;
                                    float v3 = (v1*v2 - TH*TH)/(1.0-TH*TH);
                                    atomicAdd(
                                        &outCons[zo][yo][xo][z1][y1][x1],
                                        v3);
                                    // atomicAdd(
                                    //     &outConsCnt[zo][yo][xo][z1][y1][x1],
                                    //     1);
                                }
                                // if one foreground/one background,
                                // decrease consensus
                                else if(v2 < THI) {
                                    // reverse order if pixel 2 before pixel1
                                    if(po2 <= po1) {
                                        zo = pz1-pz2;
                                        zo += PSZ-1;
                                        yo = py1-py2;
                                        yo += PSY-1;
                                        xo = px1-px2;
                                        xo += PSX-1;

                                        // atomicAdd(
                                        //     &outCons[zo][yo][xo][z2][y2][x2],
                                        //     -1);
                                        // float v3 = v1*(1-v2);
                                        float v3 = (v1*(1-v2) - TH*TH)/(1.0-TH*TH);
                                        // v3 = v3*4/3;
                                        atomicAdd(
                                            &outCons[zo][yo][xo][z2][y2][x2],
                                            -v3);
                                        // atomicAdd(
                                        //     &outConsCnt[zo][yo][xo][z2][y2][x2],
                                        //     1);
                                    }
                                    else {
                                        // atomicAdd(
                                        //     &outCons[zo][yo][xo][z1][y1][x1],
                                        //     -1);
                                        // v3 = v3*4/3;
                                        // float v3 = v1*(1-v2);
                                        float v3 = (v1*(1-v2) - TH*TH)/(1.0-TH*TH);
                                        atomicAdd(
                                            &outCons[zo][yo][xo][z1][y1][x1],
                                            -v3);
                                        // atomicAdd(
                                        //     &outConsCnt[zo][yo][xo][z1][y1][x1],
                                        //     1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// device function to set the 3D volume
__global__ void fillConsensusArray_allPatches3(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
{
    // pixel for this thread: idz, idy, idx
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    //unsigned idz = 0;

    _fillConsensusArray3(idx, idy, idz, inPred, inOverlap, outCons);
    // _fillConsensusArray(idx, idy, idz, inPred, outCons);
}


// device function to set the 3D volume
__global__ void fillConsensusArray_subsetPatches3(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const unsigned patchesIDs[], const uint64_t numPatches)
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id >= numPatches)
        return;

    int idz = patchesIDs[id*3+0];
    int idy = patchesIDs[id*3+1];
    int idx = patchesIDs[id*3+2];

    _fillConsensusArray3(idx, idy, idz, inPred, inOverlap, outCons);
    // _fillConsensusArray(idx, idy, idz, inPred, outCons);
}

#ifdef MAIN_FILLCONSENSUS

#include "verySimpleArgParse.h"
#include "cuda_vote_instances.h"

int main(int argc, char *argv[])
{
    std::string affinitiesFileName = getAndCheckArg(argc, argv,
                                                    "--affinities");
    std::string consensusFileName = getAndCheckArg(argc, argv, "--consensus");;
    predAff_t *inPredAffinitiesGPU = allocLoadPred(affinitiesFileName);
    consensus_t *outConsensusGPU = allocInitConsensus();

    computeConsensus(consensusFileName, inPredAffinitiesGPU, outConsensusGPU);
    return 0;
}
#endif
