#include <cstdint>

// sets of fg/bg pixels in python not sorted, so slightly different result
// here, total sum over array should be identical
__device__ void _fillConsensusArray(
    unsigned idx, unsigned idy, unsigned idz,
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#ifdef OVERLAP
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
#endif
#if defined OUTPUT_BOTH
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#elif defined OUTPUT_CNT
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#else
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#endif
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
#ifdef OVERLAP
                    if(inOverlap[z1][y1][x1] != 0){
                        continue;
                    }
#endif

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

#ifdef OVERLAP
                                if(inOverlap[z2][y2][x2] != 0){
                                    continue;
                                }
#endif

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
#if defined OUTPUT_CNT || defined OUTPUT_BOTH
                                    atomicAdd(
                                        &outConsCnt[zo][yo][xo][z1][y1][x1],
                                        1);
#endif

#ifndef OUTPUT_CNT
    #if defined NORM_PROB_PRODUCT
				    float v3 = (v1 * v2 - TH*TH)/(1.0-TH*TH);
    #elif defined PROB_PRODUCT
				    float v3 = v1 * v2;
#else
				    float v3 = 1;
#endif
                                    atomicAdd(
                                        &outCons[zo][yo][xo][z1][y1][x1],
                                        v3);
#endif
                                }
                                // if one foreground/one background,
                                // decrease consensus
#if defined USE_INV_TH
                                else if(v2 < THI) {
#elif defined USE_HALF_TH
                                else if(v2 < TH/2) {
#elif defined USE_LESS_THAN_TH
                                else if(v2 < TH) {
#endif

#ifndef OUTPUT_CNT
    #if defined NORM_PROB_PRODUCT
				    float v3 = (v1 * (1-v2) - TH*TH)/(1.0-TH*TH);
    #elif defined PROB_PRODUCT
				    float v3 = v1 * (1-v2);
#else
				    float v3 = 1;
#endif
#endif
                                    // reverse order if pixel 2 before pixel1
                                    if(po2 <= po1) {
                                        zo = pz1-pz2;
                                        zo += PSZ-1;
                                        yo = py1-py2;
                                        yo += PSY-1;
                                        xo = px1-px2;
                                        xo += PSX-1;

#if defined OUTPUT_CNT || defined OUTPUT_BOTH
				    atomicAdd(
					&outConsCnt[zo][yo][xo][z2][y2][x2],
                                        1);
#endif
#ifndef OUTPUT_CNT
                                    atomicAdd(
                                        &outCons[zo][yo][xo][z2][y2][x2],
                                        -v3);
#endif
                                    }
                                    else {
#if defined OUTPUT_CNT || defined OUTPUT_BOTH
				    atomicAdd(
                                        &outConsCnt[zo][yo][xo][z1][y1][x1],
                                        1);
#endif
#ifndef OUTPUT_CNT
				    atomicAdd(
                                        &outCons[zo][yo][xo][z1][y1][x1],
                                        -v3);
#endif
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
__global__ void fillConsensusArray_allPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#ifdef OVERLAP
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
#endif
#ifdef OUTPUT_BOTH
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#elif defined OUTPUT_CNT
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#else
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE])
#endif
{
    // pixel for this thread: idz, idy, idx
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    //unsigned idz = 0;

#if defined OUTPUT_BOTH
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outCons, outConsCnt);
#elif defined OUTPUT_CNT
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outConsCnt);
#else
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outCons);
#endif
}


// device function to set the 3D volume
__global__ void fillConsensusArray_subsetPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#ifdef OVERLAP
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
#endif
#ifdef OUTPUT_BOTH
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#elif defined OUTPUT_CNT
    float outConsCnt[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#else
    float outCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
#endif
    const unsigned patchesIDs[], const uint64_t numPatches)
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id >= numPatches)
        return;

    int idz = patchesIDs[id*3+0];
    int idy = patchesIDs[id*3+1];
    int idx = patchesIDs[id*3+2];

#if defined OUTPUT_BOTH
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outCons, outConsCnt);
#elif defined OUTPUT_CNT
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outConsCnt);
#else
    _fillConsensusArray(idx, idy, idz, inPred,
#ifdef OVERLAP
			inOverlap,
#endif
			outCons);
#endif
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
