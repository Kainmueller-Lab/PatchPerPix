DEFINES=-DDATAXSIZE=${DATAXSIZE}ULL -DDATAYSIZE=${DATAYSIZE}ULL \
	-DDATAZSIZE=${DATAZSIZE}ULL -DDATACSIZE=${DATACSIZE}ULL \
	-DPSX=${PSX} -DPSY=${PSY} -DPSZ=${PSZ} \
	-DNSX=${NSX} -DNSY=${NSY} -DNSZ=${NSZ} -DTH=${TH} -DTHI=${THI}

FLAGS=-O0 -lz -g -std=c++11 -lexplain -G --compiler-options -Wall
NVCC=/usr/local/cuda-10.0/bin/nvcc
FILES=cuda_vote_instances.cu \
	  fillConsensusArray.cu \
	  rankPatches.cu \
	  computePatchGraph.cu \
	  cnpy/cnpy.cpp

default:
	${NVCC} -o cuda_vote_instances main.cu ${FILES} \
	${FLAGS} ${DEFINES}

fillConsensus:
	${NVCC} -o fillConsensus ${FILES} \
	${FLAGS} ${DEFINES} -DMAIN_FILLCONSENSUS

rankPatches:
	${NVCC} -o rankPatches ${FILES} \
	${FLAGS} ${DEFINES} -DMAIN_RANKPATCHES

patchGraph:
	${NVCC} -o patchGraph ${FILES} \
	${FLAGS} ${DEFINES} -DMAIN_PATCHGRAPH
