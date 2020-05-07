#include "verySimpleArgParse.h"
#include "cuda_vote_instances.h"


int main(int argc, char *argv[])
{
	std::cout << "Start main cuda_vote_instances..." << std::endl;
	for(int i = 0; i < argc; i++) {
	 	std::cout << argv[i] << std::endl;
	}

	// parse command line arguments
	std::string affinitiesFileName =
		getAndCheckArg(argc, argv, "--affinities");
	std::string outDir =
		getAndCheckArg(argc, argv, "--result_folder") + std::string("/");
	std::string pythonVoteInstancesFileName =
		getAndCheckArg(argc, argv, "--vote_instances_path");
	bool inclSinglePatchCCS =
		std::atoi(getAndCheckArg(argc, argv, "--inclSinglePatchCCS").data());

	std::cout << "WARNING!!! set stack to unlimited with ulimit -s unlimited"
			  << std::endl;
    std::cout << "x= " << DATAXSIZE << ", y= " << DATAYSIZE
              << ", z= " << DATAZSIZE << ", c= " << DATACSIZE << std::endl;
    std::cout << "psx= " << PSX << ", psy= " << PSY << ", psz= " << PSZ
              << ", nsx= " << NSX << ", nsy= " << NSY << ", nsz= " << NSZ
              << ", th= " << TH << std::endl;

    // loading npy, init buffers
	predAff_t *inPredAffinitiesGPU = allocLoadPred(affinitiesFileName);
	scores_t *outScoreGPU = allocInitScores();
	consensus_t *outConsensusGPU = allocInitConsensus();

	// --------------------------------------------------------------------
	// (1) create consensus array
	std::string consensusFileName = outDir + std::string("consensus.bin");
	computeConsensus(inPredAffinitiesGPU, outConsensusGPU, consensusFileName);

    // --------------------------------------------------------------------
    // (2) rank patches
	std::string scoresFileName = outDir + std::string("scores.npy");
	computePatchRanking(inPredAffinitiesGPU, outConsensusGPU, outScoreGPU,
                        scoresFileName);

    // --------------------------------------------------------------------
    // (3) cover foreground --> call python vote_instances
	computeCoverFg_ext(pythonVoteInstancesFileName, affinitiesFileName,
					   scoresFileName, outDir, inclSinglePatchCCS);

	// --------------------------------------------------------------------
    // (3a) create consensus array --> only on selected patches (optional)
	bool doComputeConsensusOnSelectedPatches = false;
	if(doComputeConsensusOnSelectedPatches) {
		cudaFree(outConsensusGPU);
		cudaCheckErrors("free array consensus");
		outConsensusGPU = allocInitConsensus();
		unsigned *patchIDsGPU = nullptr;
		std::string selPatchesFileName =
			outDir + std::string("selected_patches.npy");
        unsigned numPatches = allocLoadSelectedPatchIDs(selPatchesFileName,
                                                        &patchIDsGPU);
		computeConsensusSelectedPatches(inPredAffinitiesGPU, outConsensusGPU,
										patchIDsGPU, numPatches,
										consensusFileName);
		cudaFree(patchIDsGPU);
	}


	// --------------------------------------------------------------------
    // (3b) load foreground cover
	unsigned *pairIDsGPU = nullptr;
	std::string selPatchPairsFileName =
		outDir + std::string("selected_patch_pairs.npy");
	unsigned numPatchPairs = allocLoadFgCover(selPatchPairsFileName,
											  &pairIDsGPU);


	// --------------------------------------------------------------------
	// (4) compute computePatchAffGraph
	float* patchAffGraphGPU = allocInitPatchAffGraph(numPatchPairs);
	std::string patchAffGraphFileName = outDir + std::string("affGraph.npy");
	computePatchAffGraph(inPredAffinitiesGPU, outConsensusGPU,
						 patchAffGraphGPU, pairIDsGPU, numPatchPairs,
						 patchAffGraphFileName);

    // --------------------------------------------------------------------
    // (5) get instances from patch graph
	computeLabeling_ext(pythonVoteInstancesFileName, affinitiesFileName,
						selPatchPairsFileName, patchAffGraphFileName,
						outDir);


	cudaFree(inPredAffinitiesGPU);
	cudaFree(outScoreGPU);
	cudaFree(outConsensusGPU);
	cudaFree(pairIDsGPU);
    return 0;
}
