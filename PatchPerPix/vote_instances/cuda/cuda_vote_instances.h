#ifndef CUDA_VOTE_INSTANCES_H
#define CUDA_VOTE_INSTANCES_H

// set defines in compile command
// #define DATAXSIZE 2048ULL
// #define DATAYSIZE 1024ULL
// #define DATAZSIZE 1ULL
// #define DATACSIZE 625ULL
// #define PSX 25
// #define PSY 25
// #define PSZ 1
// #define TH 0.9
// #define THI 0.1

//define the chunk sizes that each threadblock will work on
#define BLKXSIZEVOL 8
#define BLKYSIZEVOL 8
#define BLKZSIZEVOL 8

#define BLKXSIZELIST 512
#define BLKYSIZELIST 1
#define BLKZSIZELIST 1

#include <chrono>
#include <string>
#include <cstdlib>
#include <iostream>
#include <cstring>

#include "cnpy/cnpy.h"

#include "fillConsensusArray.h"
#include "rankPatches.h"
#include "computePatchGraph.h"


typedef float predAff_t[DATAZSIZE][DATAYSIZE][DATAXSIZE];
typedef float scores_t[DATAYSIZE][DATAXSIZE];
typedef float consensus_t[NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE];


// for cuda error checking
#define cudaCheckErrors(msg)      \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
                    std::exit(1); \
        } \
    } while (0)


// cuda kernel dimensions
// heads-up: dimension order changed here
dim3 getBlockSizeVol();
dim3 getGridSizeVol();

std::chrono::time_point<std::chrono::high_resolution_clock> preKernel(std::string name);
void postKernel(
    std::string name,
    std::chrono::time_point<std::chrono::high_resolution_clock> tBefore);

predAff_t* allocLoadPred(std::string affinitiesFileName);
scores_t* allocInitScores();
consensus_t* allocConsensus();
consensus_t* allocInitConsensus();
consensus_t* allocLoadConsensus(std::string consensusFileName);
unsigned allocLoadSelectedPatchIDs(std::string selPatchesFileName,
                                   unsigned** patchIDsGPU);
unsigned allocLoadFgCover(std::string selPatchPairsFileName,
                          unsigned **pairIDsGPU);
float *allocInitPatchAffGraph(unsigned numPatchPairs);

void writeConsensus(consensus_t *outConsensusGPU, std::string fileName);
void writeScores(scores_t *outScoreGPU, std::string fileName);
void writePatchGraph(float *patchAffGraphGPU, unsigned numPatchPairs,
                     std::string fileName);

void computeConsensus(predAff_t *inPredAffinitiesGPU,
                      consensus_t *outConsensusGPU,
                      std::string fileName,
                      bool doWriteConsensus = true);

void computeConsensusSelectedPatches(predAff_t *inPredAffinitiesGPU,
                                     consensus_t *outConsensusGPU,
                                     unsigned *patchIDsGPU,
                                     unsigned numPatches,
                                     std::string fileName,
                                     bool doWriteConsensus = false);

void computePatchRanking(predAff_t *inPredAffinitiesGPU,
                         consensus_t *inConsensusGPU,
                         scores_t *outScoreGPU,
                         std::string fileName);

void computeCoverFg_ext(std::string pythonVoteInstancesFileName,
                        std::string affinitiesFileName,
                        std::string scoresFileName,
                        std::string outDir,
                        bool inclSinglePatchCCS = true);

void computePatchAffGraph(predAff_t *inPredAffinitiesGPU,
                          consensus_t *outConsensusGPU,
                          float *patchAffGraphGPU, unsigned *pairIDsGPU,
                          unsigned numPatchPairs, std::string fileName);

void computeLabeling_ext(std::string pythonVoteInstancesFileName,
                         std::string affinitiesFileName,
                         std::string selPatchPairsFileName,
                         std::string patchAffGraphFileName,
                         std::string outDir);


#endif /* CUDA_VOTE_INSTANCES_H */
