extern "C" __global__ void addForces(const real* __restrict__ grads,
                                     long long* __restrict__ forceBuffers,
                                     int* __restrict__ atomIndex,
                                     int gradSign,
                                     int numAtoms,
                                     int paddedNumAtoms) {
  for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < numAtoms; atom += blockDim.x * gridDim.x) {
    int index = atomIndex[atom];
    forceBuffers[atom + 0 * paddedNumAtoms] -= gradSign * (long long)(grads[3 * index + 0] * 0x100000000);
    forceBuffers[atom + 1 * paddedNumAtoms] -= gradSign * (long long)(grads[3 * index + 1] * 0x100000000);
    forceBuffers[atom + 2 * paddedNumAtoms] -= gradSign * (long long)(grads[3 * index + 2] * 0x100000000);
  }
}

extern "C" __global__ void copyInputs(real* __restrict__ posTensor,
                                      real* __restrict__ boxTensor,
                                      const real4* __restrict__ posq,
                                      int* __restrict__ atomIndex,
                                      int numAtoms,
                                      real4 periodicBoxVecX,
                                      real4 periodicBoxVecY,
                                      real4 periodicBoxVecZ) {
  for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < numAtoms; atom += blockDim.x * gridDim.x) {
    real4 pos = posq[atom];
    int index = atomIndex[atom];
    posTensor[3 * index + 0] = pos.x;
    posTensor[3 * index + 1] = pos.y;
    posTensor[3 * index + 2] = pos.z;
  }
  if (blockIdx.x == 0 and threadIdx.x == 0) {
    boxTensor[0] = periodicBoxVecX.x;
    boxTensor[1] = periodicBoxVecX.y;
    boxTensor[2] = periodicBoxVecX.z;
    boxTensor[3] = periodicBoxVecY.x;
    boxTensor[4] = periodicBoxVecY.y;
    boxTensor[5] = periodicBoxVecY.z;
    boxTensor[6] = periodicBoxVecZ.x;
    boxTensor[7] = periodicBoxVecZ.y;
    boxTensor[8] = periodicBoxVecZ.z;
  }
}
