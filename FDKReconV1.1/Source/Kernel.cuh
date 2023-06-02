#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <Windows.h>

#include <cmath>
#include <iostream>


#include "CBCTFDKRecon.h"

#define CUDAFREE(varP)\
 if(varP != nullptr) \
{ \
cudaFree(varP); \
 varP = nullptr;\
}

#define PI acosf(-1)

#define BLOCKSIZEX 8
#define BLOCKSIZEY 8
#define BLOCKSIZEZ 8

typedef unsigned char uchar;

struct ImagingSystemInfo;
struct ReconInfoData;
struct GeometryPara;

void chooseGPU(uchar index);

void computeDetectorPoints(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& d_mReconInfoData);
void filterFT(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& h_mReconInfoData, ReconInfoData& d_mReconInfoData);

void reconGPU(ImagingSystemInfo mImagingSystemInfo, ReconInfoData h_mReconInfoData, ReconInfoData d_mReconInfoData, GeometryPara mGeometryPara);
void prepareReconVariables(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& d_mReconInfoData);

void createTexture2D(cudaTextureObject_t& texObj, cudaArray_t& cuArray, float* data, size_t width, size_t height);

void getCurrentCursorCoordinate(HANDLE & hConsole, COORD & coord);


__global__ void computeDetecPointsCoorsKernel(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData);
__global__ void filterAmplitude(cufftComplex* d_filter, size_t dNum);
__global__ void computeImgCoordinates(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData);
__global__ void weightProjection(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData);
__global__ void projectionFilterInTDF(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData);
__global__ void processFilteredProjIfft(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData);
__global__ void reconstructeImage(float angle, cudaTextureObject_t texProj, ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData, GeometryPara mGeometryPara);






