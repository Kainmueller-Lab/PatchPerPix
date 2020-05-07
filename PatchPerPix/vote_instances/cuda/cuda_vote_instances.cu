#include "cuda_vote_instances.h"

// cuda kernel dimensions
// heads-up: dimension order changed here
dim3 getBlockSizeVol() {
    return dim3(BLKXSIZEVOL, BLKYSIZEVOL, BLKZSIZEVOL);
}
dim3 getGridSizeVol() {
    dim3 gridSize(((DATAXSIZE+BLKXSIZEVOL-1)/BLKXSIZEVOL),
                  ((DATAYSIZE+BLKYSIZEVOL-1)/BLKYSIZEVOL),
                  ((DATAZSIZE+BLKZSIZEVOL-1)/BLKZSIZEVOL));
    std::cout << "grid size (for volume): " << gridSize.x << " "
              << gridSize.y << " " << gridSize.z << std::endl;

    return gridSize;
}

std::chrono::time_point<std::chrono::high_resolution_clock> preKernel(std::string name) {
    cudaDeviceSynchronize();
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::cout << name << " start" << std::endl;
    return tBefore;
}

void postKernel(
    std::string name,
    std::chrono::time_point<std::chrono::high_resolution_clock> tBefore) {
    cudaDeviceSynchronize();
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::string tmp = std::string("Kernel ") + name +
        std::string(" launch failure");
    cudaCheckErrors(tmp.data());
    std::cout << name << " end" << std::endl;
    std::cout << "time " << name << ": "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

}

predAff_t* allocLoadPred(std::string affinitiesFileName) {
    predAff_t *inPredAffinitiesGPU;
    auto tBefore = std::chrono::high_resolution_clock::now();
    auto predAffinitiesNpy = cnpy::npy_load(affinitiesFileName);
    float* predAffinitiesTmp = predAffinitiesNpy.data<float>();
    float (&predAffinities)[DATACSIZE][DATAZSIZE][DATAYSIZE][DATAXSIZE] =
        *reinterpret_cast<float (*)[DATACSIZE][DATAZSIZE][DATAYSIZE][DATAXSIZE]>(predAffinitiesTmp);

    cudaMallocManaged(&inPredAffinitiesGPU,
                      DATAXSIZE*DATAYSIZE*DATAZSIZE*DATACSIZE*sizeof(float));
    cudaCheckErrors("alloc array affinities");
    std::memcpy(inPredAffinitiesGPU, predAffinities,
                DATAXSIZE*DATAYSIZE*DATAZSIZE*DATACSIZE*sizeof(float));

    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time loading affinities: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return inPredAffinitiesGPU;
}

scores_t* allocInitScores() {
    scores_t *outScoreGPU;
    auto tBefore = std::chrono::high_resolution_clock::now();
    cudaMallocManaged(&outScoreGPU,
                      DATAXSIZE*DATAYSIZE*DATAZSIZE*sizeof(float));
    cudaCheckErrors("alloc array scores");
    std::memset(outScoreGPU, 0,
                DATAXSIZE*DATAYSIZE*DATAZSIZE*sizeof(float));
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time init scores: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return outScoreGPU;
}

consensus_t* allocConsensus() {
    consensus_t *outConsensusGPU;
    cudaMallocManaged(
        &outConsensusGPU,
        DATAXSIZE*DATAYSIZE*DATAZSIZE*NSX*NSY*NSZ*sizeof(float));
    cudaCheckErrors("alloc array consensus");
    return outConsensusGPU;
}
consensus_t* allocInitConsensus() {
    consensus_t *outConsensusGPU;
    auto tBefore = std::chrono::high_resolution_clock::now();
    outConsensusGPU = allocConsensus();
    std::memset(
        outConsensusGPU, 0,
        DATAXSIZE*DATAYSIZE*DATAZSIZE*NSX*NSY*NSZ*sizeof(float));
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time init consensus: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return outConsensusGPU;
}

consensus_t* allocLoadConsensus(std::string consensusFileName) {
    consensus_t *outConsensusGPU;
    auto tBefore = std::chrono::high_resolution_clock::now();
    outConsensusGPU = allocConsensus();
    FILE* fp = fopen(consensusFileName.data(), "rb");
    if (fp == nullptr) {
        std::cerr << "cannot read file " << consensusFileName << std::endl;
        std::exit(1);
    }
    fseek (fp , 0 , SEEK_END);
    uint64_t sz = ftell (fp)/4;
    rewind (fp);
    uint64_t result = fread (outConsensusGPU, 4, sz, fp);
    if (result != sz) {
        std::cerr << "reading error " << consensusFileName
                  << "(" << result << "/" << sz << ")" << std::endl;
        std::exit(1);
    }
    fclose(fp);
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time load consensus: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return outConsensusGPU;
}

unsigned allocLoadSelectedPatchIDs(std::string selPatchesFileName,
                                   unsigned** patchIDsGPU) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    auto patchIDsNpy = cnpy::npy_load(selPatchesFileName);
    unsigned* patchIDs = patchIDsNpy.data<unsigned>();
    const uint64_t numPatches = uint64_t(patchIDsNpy.shape[0]);
    std::cout << "load selected patches: " << numPatches << " "
              << patchIDsNpy.shape[0] << " " << patchIDsNpy.shape[1]
              << std::endl;

    cudaMallocManaged(patchIDsGPU, 3*numPatches*sizeof(unsigned));
    cudaCheckErrors("alloc array patch ids");
    std::memcpy(*patchIDsGPU, patchIDs, 3*numPatches*sizeof(unsigned));

    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time loading patch ids: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return numPatches;
}

unsigned allocLoadFgCover(std::string selPatchPairsFileName,
                          unsigned **pairIDsGPU) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    auto patchPairsNpy = cnpy::npy_load(selPatchPairsFileName);
    unsigned* patchPairs = patchPairsNpy.data<unsigned>();
    const uint64_t numPatchPairs = uint64_t(patchPairsNpy.shape[0]);
    std::cout << "load selected patch pairs: " << numPatchPairs << " "
              << patchPairsNpy.shape[0] << " " << patchPairsNpy.shape[1]
              << std::endl;
    if(numPatchPairs > 2147483647) {
        std::cout << "too many pairs" << std::endl;
        std::exit(1);
    }

    cudaMallocManaged(pairIDsGPU, 6 * numPatchPairs * sizeof(unsigned));
    cudaCheckErrors("alloc array pair ids fg cover");
    std::memcpy(*pairIDsGPU, patchPairs, 6 * numPatchPairs * sizeof(unsigned));

    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time cover load: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return numPatchPairs;
}

float *allocInitPatchAffGraph(unsigned numPatchPairs) {
    float *patchAffGraphGPU;
    auto tBefore = std::chrono::high_resolution_clock::now();
    cudaMallocManaged(&patchAffGraphGPU, numPatchPairs * sizeof(float));
    cudaCheckErrors("alloc array patch aff graph");
    std::memset(patchAffGraphGPU, 0, numPatchPairs * sizeof(unsigned));
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time init patchAffGraph: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;

    return patchAffGraphGPU;
}

void writeConsensus(consensus_t *outConsensusGPU, std::string fileName) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::vector<std::uint64_t> shapeConsensus;
    shapeConsensus.push_back((NSZ));
    shapeConsensus.push_back((NSY));
    shapeConsensus.push_back((NSX));
    shapeConsensus.push_back((DATAZSIZE));
    shapeConsensus.push_back((DATAYSIZE));
    shapeConsensus.push_back((DATAXSIZE));
    int32_t outConsensus[NSZ][NSY][NSX][DATAZSIZE][DATAYSIZE][DATAXSIZE];
    float maxC = -999999;
    float minC = 999999;
    std::uint64_t sum = 0;
    for (unsigned p1 = 0; p1 < NSZ; p1++)
        for (unsigned p2 = 0; p2 < NSY; p2++)
            for (unsigned p3 = 0; p3 < NSX; p3++)
                for (unsigned i = 0; i < DATAZSIZE; i++)
                    for (unsigned j = 0; j<DATAYSIZE; j++)
                        for (unsigned k = 0; k<DATAXSIZE; k++){
                            int32_t tmp =
                                outConsensusGPU[p1][p2][p3][i][j][k];
                            outConsensus[p1][p2][p3][i][j][k] = tmp;
                            sum += tmp;
                            if(tmp > maxC)
                                maxC = tmp;
                            if(tmp < minC)
                                minC = tmp;
                        }
    std::cout << "shape consensus ";
    for(unsigned int i = 0; i < shapeConsensus.size(); i++) {
        std::cout << shapeConsensus[i] << " ";
    } std::cout << std::endl;
    std::cout << "sum/min/max consensus " << sum << " "
              << minC << " " << maxC << std::endl;
    // std::cout << outCons[49][49][DATAYSIZE-1][DATAXSIZE-1] << std::endl;
    FILE* fp = NULL;
    fp = fopen(fileName.data(), "wb");
    fwrite(outConsensus, sizeof(float),
           DATAXSIZE*DATAYSIZE*DATAZSIZE*NSX*NSY*NSZ, fp);
    fclose(fp);
    // cnpy::npy_save(fn, (int32_t*)&out, shape);
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time consensus save: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count()
              << std::endl;
}

void writeScores(scores_t *outScoreGPU, std::string fileName) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::vector<std::uint64_t> shapeScores;
    shapeScores.push_back((DATAZSIZE));
    shapeScores.push_back((DATAYSIZE));
    shapeScores.push_back((DATAXSIZE));
    int32_t outScores[DATAZSIZE][DATAYSIZE][DATAXSIZE];

    float maxS = -9999999;
    float minS = 9999999;
    for (unsigned i = 0; i < DATAZSIZE; i++) {
        for (unsigned j = 0; j < DATAYSIZE; j++) {
            for (unsigned k = 0; k < DATAXSIZE; k++) {
                outScores[i][j][k] = int32_t(outScoreGPU[i][j][k]);
                if(outScores[i][j][k] > maxS)
                    maxS = outScores[i][j][k];
                if(outScores[i][j][k] < minS)
                    minS = outScores[i][j][k];
            }
        }
    }

    cnpy::npy_save(fileName, (int32_t*)&outScores, shapeScores);
    std::cout << "Min: " << minS << " , Max: " << maxS << std::endl;
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time score save: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;
}

void writePatchGraph(float *patchAffGraphGPU, unsigned numPatchPairs,
                     std::string fileName) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::vector<std::uint64_t> shapePatchGraph;
    shapePatchGraph.push_back(numPatchPairs);
    int32_t outPatchAffGraph[numPatchPairs];
    for (unsigned i = 0; i < numPatchPairs; i++) {
        outPatchAffGraph[i] = int32_t(patchAffGraphGPU[i]);
    }
    cnpy::npy_save(fileName, (int32_t*)&outPatchAffGraph,
                   shapePatchGraph);
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time patchGraph save: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;
}


void computeConsensus(predAff_t *inPredAffinitiesGPU,
                      consensus_t *outConsensusGPU,
                      std::string fileName,
                      bool doWriteConsensus) {
    auto tBefore = preKernel("consensus");
    fillConsensusArray_allPatches<<<getGridSizeVol(),getBlockSizeVol()>>>(
        inPredAffinitiesGPU, outConsensusGPU);
    postKernel("consensus", tBefore);

    if(doWriteConsensus) {
        writeConsensus(outConsensusGPU, fileName);
    }
}

void computeConsensusSelectedPatches(predAff_t *inPredAffinitiesGPU,
                                     consensus_t *outConsensusGPU,
                                     unsigned *patchIDsGPU,
                                     unsigned numPatches,
                                     std::string fileName,
                                     bool doWriteConsensus) {
    const dim3 blockSizeList(BLKXSIZELIST);
    const dim3 gridSizeList(((numPatches+BLKXSIZELIST-1)/BLKXSIZELIST));
    std::cout << "grid size (for list): " << gridSizeList.x << std::endl;


    auto tBefore = preKernel("consensus2");
    fillConsensusArray_subsetPatches<<<gridSizeList,blockSizeList>>>(
        inPredAffinitiesGPU, outConsensusGPU, patchIDsGPU,
        numPatches);
    postKernel("consensusSelectedPatches", tBefore);

    if(doWriteConsensus) {
        writeConsensus(outConsensusGPU, fileName);
    }
}

void computePatchRanking(predAff_t *inPredAffinitiesGPU,
                         consensus_t *inConsensusGPU,
                         scores_t *outScoreGPU,
                         std::string fileName) {
    auto tBefore = preKernel("scores");
    rankPatches<<<getGridSizeVol(),getBlockSizeVol()>>>(
        inPredAffinitiesGPU, inConsensusGPU, outScoreGPU);
    postKernel("rankPatches", tBefore);

    // copy output data back to host
    writeScores(outScoreGPU, fileName);
}

void computeCoverFg_ext(std::string pythonVoteInstancesFileName,
                        std::string affinitiesFileName,
                        std::string scoresFileName,
                        std::string outDir,
                        bool inclSinglePatchCCS) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::string cmdCoverFg = std::string("python ")
        + pythonVoteInstancesFileName
        + std::string(" --affinities ") + affinitiesFileName
        + std::string(" --result_folder " ) + outDir
        + std::string(" --scores ") + scoresFileName
        + std::string(" --skipThinCover")
        + std::string(" --skipConsensus")
        + std::string(" --skipLookup")
        + std::string(" -p ") + std::to_string(PSZ)
        + std::string(" -p ") + std::to_string(PSY)
        + std::string(" -p ") + std::to_string(PSX)
        + std::string(" --termAfterThinCover");
    if (inclSinglePatchCCS) {
        cmdCoverFg += std::string(" --includeSinglePatchCCS");
    }

    std::cout << "cover foreground: " << cmdCoverFg << std::endl;
    std::system(cmdCoverFg.data());
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time cover: "
              << std::chrono::duration_cast<std::chrono::seconds>(
        tAfter-tBefore).count() << std::endl;
}

void computePatchAffGraph(predAff_t *inPredAffinitiesGPU,
                          consensus_t *outConsensusGPU,
                          float *patchAffGraphGPU, unsigned *pairIDsGPU,
                          unsigned numPatchPairs, std::string fileName) {
    const dim3 blockSizeList(BLKXSIZELIST);
    const dim3 gridSizeList(((numPatchPairs+BLKXSIZELIST-1)/BLKXSIZELIST));
    std::cout << "grid size (for list): " << gridSizeList.x << std::endl;

    auto tBeforeSelect = preKernel("computePatchGraph");
    computePatchGraph<<<gridSizeList,blockSizeList>>>(
        inPredAffinitiesGPU, outConsensusGPU, patchAffGraphGPU, pairIDsGPU,
        numPatchPairs);
    postKernel("computePatchGraph", tBeforeSelect);

    writePatchGraph(patchAffGraphGPU, numPatchPairs, fileName);
}

void computeLabeling_ext(std::string pythonVoteInstancesFileName,
                         std::string affinitiesFileName,
                         std::string selPatchPairsFileName,
                         std::string patchAffGraphFileName,
                         std::string outDir) {
    auto tBefore = std::chrono::high_resolution_clock::now();
    std::string cmdComputeSeg = std::string("python ")
        + pythonVoteInstancesFileName
        + std::string(" --affinities ") + affinitiesFileName
        + std::string(" --result_folder " ) + outDir
        + std::string(" --graphToInst")
        + std::string(" --selected_patch_pairs ") + selPatchPairsFileName
        + std::string(" -p ") + std::to_string(PSZ)
        + std::string(" -p ") + std::to_string(PSY)
        + std::string(" -p ") + std::to_string(PSX)
        + std::string(" --affgraph ") + patchAffGraphFileName;

    std::cout << "get instances from patch graph: " << cmdComputeSeg
              << std::endl;
    std::system(cmdComputeSeg.data());
    auto tAfter = std::chrono::high_resolution_clock::now();
    std::cout << "time labeling: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                  tAfter-tBefore).count() << std::endl;
}
