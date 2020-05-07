#include <cstdint>

// sets of fg/bg pixels in python not sorted, so slightly different result
// here, total sum over array should be identical
__device__ void _normConsensusArray(
    unsigned idx, unsigned idy, unsigned idz,
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][DATAYSIZE][DATAXSIZE][NSZ][NSY][NSX],
    float outConsCnt[][DATAYSIZE][DATAXSIZE][NSZ][NSY][NSX])
{
    if ((idx < (DATAXSIZE)) &&
        (idy < (DATAYSIZE)) &&
        (idz < (DATAZSIZE))){

        unsigned int mid = int((PSX*PSY*PSZ)/2);
        if(inPred[mid][idz][idy][idx] <= TH)
            return;

    for(int z = 0; z < NSZ; z++) {
        for(int y = 0; y < NSY; y++) {
            for(int x = 0; x < NSX; x++) {
	      // if(outConsCnt[z][y][x][idz][idy][idx] != 0)
	      // outCons[z][y][x][idz][idy][idx] = outCons[z][y][x][idz][idy][idx]/outConsCnt[z][y][x][idz][idy][idx];
	      if(outConsCnt[idz][idy][idx][z][y][x] != 0)
		outCons[idz][idy][idx][z][y][x] = outCons[idz][idy][idx][z][y][x]/outConsCnt[idz][idy][idx][z][y][x];
            }
        }
    }
    }
}


// device function to set the 3D volume
__global__ void normConsensusArray(
    const float inPred[][DATAZSIZE][DATAYSIZE][DATAXSIZE],
    float outCons[][DATAYSIZE][DATAXSIZE][NSZ][NSY][NSX],
    float outConsCnt[][DATAYSIZE][DATAXSIZE][NSZ][NSY][NSX])
{
  // pixel for this thread: idz, idy, idx
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

  _normConsensusArray(idx, idy, idz, inPred, outCons, outConsCnt);
}
