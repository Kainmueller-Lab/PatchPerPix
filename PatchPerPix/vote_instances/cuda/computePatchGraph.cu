#include <cstdint>

__global__ void computePatchGraph(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    const float inCons[][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float affGraph[], const unsigned pairsIDs[], const uint64_t numPairs,
    int offset)
{
    uint64_t id1 = blockIdx.x*blockDim.x + threadIdx.x;

    if(id1 >= numPairs)
        return;

    id1 += offset;

    int idz = pairsIDs[id1*6];
    int idy = pairsIDs[id1*6+1];
    int idx = pairsIDs[id1*6+2];

    int idz2 = pairsIDs[id1*6+3];
    int idy2 = pairsIDs[id1*6+4];
    int idx2 = pairsIDs[id1*6+5];

    uint32_t rnd =
        uint32_t(idz)*uint32_t(idz2)*
        uint32_t(idy)*uint32_t(idy2)*
        uint32_t(idx)*uint32_t(idx2);

    unsigned int mid = int((PSX*PSY*PSZ)/2);
    unsigned const PSXH = int(PSX/2);
    unsigned const PSYH = int(PSY/2);
    unsigned const PSZH = int(PSZ/2);

    float acc = 0.0f;
    unsigned int fgCnt = 0;

    // iterate over all pixel in patch
    for(int pz1 = 0; pz1 < PSZ; pz1++) {
        for(int py1 = 0; py1 < PSY; py1++) {
            for(int px1 = 0; px1 < PSX; px1++) {
                const int z1 = idz+pz1-PSZH;
                const int y1 = idy+py1-PSYH;
                const int x1 = idx+px1-PSXH;
                if(inPred[mid][z1][y1][x1] <= TH) {
                    continue;
                }
                int po1 = px1 + PSX * py1 + PSX * PSY * pz1;

                // if pred affinity in patch smaller than threshold, continue
                if(inPred[po1][idz][idy][idx] <= TH) {
                    continue;
                }

                for(int pz2 = 0; pz2 < PSZ; pz2++) {
                    for(int py2 = 0; py2 < PSY; py2++) {
                        for(int px2 = 0; px2 < PSX; px2++) {
                            const int z2 = idz2+pz2-PSZH;
                            const int y2 = idy2+py2-PSYH;
                            const int x2 = idx2+px2-PSXH;
                            if(inPred[mid][z2][y2][x2] <= TH) {
                                continue;
                            }
                            int po2 = px2 + PSX * py2 + PSX * PSY * pz2;
                            if(inPred[po2][idz2][idy2][idx2] <= TH) {
                                continue;
                            }

                            int gz1 = x1 + DATAXSIZE * y1 +
                                DATAXSIZE * DATAYSIZE * z1;
                            // bug? was idx2+py2
                            int gz2 = x2 + DATAXSIZE * y2 +
                                DATAXSIZE * DATAYSIZE * (z2);

                            // intersection
                            if (abs(int(x1-idx2)) <= PSXH &&
                                abs(int(y1-idy2)) <= PSYH &&
                                abs(int(z1-idz2)) <= PSZH &&
                                abs(int(x2-idx)) <= PSXH &&
                                abs(int(y2-idy)) <= PSYH &&
                                abs(int(z2-idz)) <= PSZH)
                            {
                                rnd = rnd*1103515245U;
                                float rndT = rnd/4294967296.0f;
                                if (rndT > 0.2)
                                    continue;
                            }


                            if(gz1 <= gz2) {
                                int zo = idz2+pz2-idz-pz1;
                                int yo = idy2+py2-idy-py1;
                                int xo = idx2+px2-idx-px1;

                                zo += PSZ-1;
                                yo += PSY-1;
                                xo += PSX-1;

                                if(zo < 0 || zo >= 2*PSZ ||
                                   yo < 0 || yo >= 2*PSY ||
                                   xo < 0 || xo >= 2*PSX)
                                    continue;

                                float v3 = inCons[zo][yo][xo][z1][y1][x1];
                                acc += v3;
                                fgCnt += 1;
                            }
                            else if(gz2 < gz1) {
                                int zo = idz+pz1-idz2-pz2;
                                int yo = idy+py1-idy2-py2;
                                int xo = idx+px1-idx2-px2;

                                zo += PSZ-1;
                                yo += PSY-1;
                                xo += PSX-1;

                                if(zo < 0 || zo >= 2*PSZ ||
                                   yo < 0 || yo >= 2*PSY ||
                                   xo < 0 || xo >= 2*PSX)
                                    continue;

                                float v3 = inCons[zo][yo][xo][z2][y2][x2];
                                acc += v3;
                                fgCnt += 1;
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef NORM_PATCH_AFFINITY
    affGraph[id1] = acc/float(max(1, fgCnt));
#else
    affGraph[id1] = acc;
#endif
}
#ifdef MAIN_PATCHGRAPH

#include "verySimpleArgParse.h"
#include "cuda_vote_instances.h"

int main(int argc, char *argv[])
{
    std::string affinitiesFileName = getAndCheckArg(argc, argv,
                                                    "--affinities");
    std::string consensusFileName = getAndCheckArg(argc, argv, "--consensus");;
    std::string selPatchesFileName = getAndCheckArg(argc, argv,
                                                    "--selected_patches");
    std::string patchAffGraphFileName = getAndCheckArg(argc, argv,
                                                       "--affGraph");
    predAff_t *inPredAffinitiesGPU = allocLoadPred(affinitiesFileName);
    consensus_t *inConsensusGPU = allocLoadConsensus(consensusFileName);
    unsigned *pairIDsGPU = nullptr;
    unsigned numPatchPairs = allocLoadFgCover(selPatchesFileName, pairIDsGPU);
    float* patchAffGraphGPU = allocInitPatchAffGraph(numPatchPairs);

    computePatchAffGraph(patchAffGraphFileName, inPredAffinitiesGPU,
                         inConsensusGPU, patchAffGraphGPU, pairIDsGPU,
                         numPatchPairs);
    return 0;
}
#endif
