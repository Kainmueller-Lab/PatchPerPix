__global__ void rankPatches(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const float inCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const bool inOverlap[DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outScore[][DATAYSIZE][DATAXSIZE])
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

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

        if(inPred[mid][idz][idy][idx] <= TH)
            return;

        float acc = 0.0f;
        unsigned int fgCnt = 0;

        // for all pixels in patch
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

                                if (po1 == po2) {
                                    continue;
                                }
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
                                // if both high affinity, increase acc
                                if(v2 > TH) {
                                    if(po2 <= po1)
                                        continue;

                                    float v3 = inCons[zo][yo][xo][z1][y1][x1];
#ifdef COUNT_POS_NEG
				    if (v3 != 0)
                                        acc += copysignf(1, v3);
                                    else
                                        acc -= 1;
#else
                                    acc += v3;
#endif
                                }
#if defined USE_INV_TH
                                else if(v2 < THI) {
#elif defined USE_HALF_TH
                                else if(v2 < TH/2) {
#elif defined USE_LESS_THAN_TH
                                else if(v2 < TH) {
#endif
                                    if(po2 <= po1) {
                                        zo = pz1-pz2;
                                        zo += PSZ-1;
                                        yo = py1-py2;
                                        yo += PSY-1;
                                        xo = px1-px2;
                                        xo += PSX-1;

                                        float v3 = inCons[zo][yo][xo][z2][y2][x2];
#ifdef COUNT_POS_NEG
                                        if (v3 != 0)
                                            acc -= copysignf(1, v3);
                                        else
                                            acc -= 1;
#else
                                        acc -= v3;
#endif
                                    }
                                    else {
                                        float v3 = inCons[zo][yo][xo][z1][y1][x1];
#ifdef COUNT_POS_NEG
                                        if (v3 != 0)
                                            acc -= copysignf(1, v3);
                                        else
                                            acc -= 1;
#else
                                        acc -= v3;
#endif
                                    }
                                }
                                fgCnt += 1;
                            }
                        }
                    }
                }
            }
        }
#ifdef NORM_PATCH_RANK
        outScore[idz][idy][idx] = acc/float(max(1, fgCnt));
#else
        outScore[idz][idy][idx] = acc;
#endif
    }
    else if ((idx < DATAXSIZE) &&
             (idy < DATAYSIZE) &&
             (idz < DATAZSIZE)) {
#ifdef NORM_PATCH_RANK
        outScore[idz][idy][idx] = -1.0;
#else
        outScore[idz][idy][idx] = -9999999.0;
#endif
    }
}

#ifdef MAIN_RANKPATCHES

#include "verySimpleArgParse.h"
#include "cuda_vote_instances.h"

int main(int argc, char *argv[])
{
    std::string affinitiesFileName = getAndCheckArg(argc, argv,
                                                    "--affinities");
    std::string consensusFileName = getAndCheckArg(argc, argv, "--consensus");;
    std::string scoresFileName = getAndCheckArg(argc, argv, "--scores");

    predAff_t *inPredAffinitiesGPU = allocLoadPred(affinitiesFileName);
    scores_t *outScoreGPU = allocInitScores();
    consensus_t *inConsensusGPU = allocLoadConsensus(consensusFileName);

    computePatchRanking(scoresFileName, inPredAffinitiesGPU, inConsensusGPU,
                        outScoreGPU);
    return 0;
}
#endif
