#include "Kernel.cuh"

#include <stdio.h>


void reconGPU(ImagingSystemInfo mImagingSystemInfo, ReconInfoData h_mReconInfoData, ReconInfoData d_mReconInfoData, GeometryPara mGeometryPara)
{
	//chooseGPU(0);
	cudaError_t cudaStatus;
	
	computeDetectorPoints(mImagingSystemInfo, d_mReconInfoData);

	filterFT(mImagingSystemInfo, h_mReconInfoData, d_mReconInfoData);

#if 0
	cudaStatus = cudaMemcpy(h_mReconInfoData.filter, d_mReconInfoData.filter, mImagingSystemInfo.dNumU * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable h_mReconInfoData.filter cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
		exit(0);
	}

	for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		std::cout << h_mReconInfoData.filter[i].x << " " << std::endl;
	}
#endif

	prepareReconVariables(mImagingSystemInfo, d_mReconInfoData);

#if 0
	cudaStatus = cudaMemcpy(d_mReconInfoData.detTempV, d_mReconInfoData.x, mImagingSystemInfo.pNumX * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable h_mReconInfoData.detTempV cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
		exit(0);
	}

	for (int i = 0; i < mImagingSystemInfo.dNumV; i++)
	{
		std::cout << d_mReconInfoData.detTempV[i] << " " << std::endl;
	}
#endif


	dim3 blockSizeRec(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
	dim3 gridSizeRec((mImagingSystemInfo.pNumX - 1) / blockSizeRec.x + 1, (mImagingSystemInfo.pNumY - 1) / blockSizeRec.y + 1, (mImagingSystemInfo.pNumZ - 1) / blockSizeRec.z + 1);

	dim3 blockSizeWProj(BLOCKSIZEX, BLOCKSIZEY);
	dim3 gridSizeWProj((mImagingSystemInfo.dNumU - 1) / blockSizeWProj.x + 1, (mImagingSystemInfo.dNumV - 1) / blockSizeWProj.y + 1);

	// ����ͶӰ����FFT cufftPlan1d
	cufftHandle planProj;                           // Create cuda library function handle
	cufftPlan1d(&planProj, mImagingSystemInfo.dNumU, CUFFT_C2C, mImagingSystemInfo.dNumV);    // Plan declaration

	// �����ؽ������2D����ͶӰ����
	cudaArray_t cuArray;
	cudaTextureObject_t texObj = 0;
	createTexture2D(texObj, cuArray, d_mReconInfoData.filterProj, mImagingSystemInfo.dNumU, mImagingSystemInfo.dNumV);


	// ----------------------------��Ƕ��ؽ�
	std::cout << "�ؽ���ʼ ..." << std::endl;
	float angle = 0.0f;
	// Timing


	std::cout << "����� ";
	HANDLE hConsole;
	COORD coord;
	getCurrentCursorCoordinate(hConsole, coord);

	cudaEvent_t g_start, g_stop;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	cudaEventRecord(g_start, 0);

	for (size_t i = 0; i < mImagingSystemInfo.views; i++)        
	{
		//std::cout << "�� " << i + 1 << " ���Ƕ�" << "==>>" << std::endl;

		// ��ȡһ���Ƕȵ�ͶӰ
		for (size_t j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			h_mReconInfoData.proj[j] = h_mReconInfoData.totalProj[j + i * mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
		}


		cudaStatus = cudaMemcpy(d_mReconInfoData.proj, h_mReconInfoData.proj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Variable d_mReconInfoData.proj cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}
#if 0
		for (size_t j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			std::cout << h_mReconInfoData.proj[j] << " ";
		}
#endif
		// 1. weight the projection
		weightProjection << <gridSizeWProj, blockSizeWProj >> > (mImagingSystemInfo, d_mReconInfoData);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "weightKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching weightKernel!\n", cudaStatus);
			exit(0);
		}

#if 0
		cudaStatus = cudaMemcpy(h_mReconInfoData.weightProj, d_mReconInfoData.weightProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Variable d_mReconInfoData.proj cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}

		for (size_t j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			std::cout << h_mReconInfoData.weightProj[j].x << " ";
		}
#endif


		// 2. Filter projections in the frequency domain
		// ͶӰ����FFT
		//cudaMalloc(&d_mReconInfoData.wfProj, )
		cufftExecC2C(planProj, (cufftComplex*)d_mReconInfoData.weightProj, (cufftComplex*)d_mReconInfoData.weightProj, CUFFT_FORWARD);  //execute FFT

#if 0
		cudaStatus = cudaMemcpy(h_mReconInfoData.weightProj, d_mReconInfoData.weightProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Variable h_mReconInfoData.filter cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}

		for (int j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			std::cout << h_mReconInfoData.weightProj[j].x << " + " << h_mReconInfoData.weightProj[j].y << " i" << "  ";
		}
#endif

		// Ƶ����ͶӰ�����˲�
		projectionFilterInTDF << <gridSizeWProj, blockSizeWProj >> > (mImagingSystemInfo, d_mReconInfoData);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "projectionFilterInTDFKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching projectionFilterInTDFKernel!\n", cudaStatus);
			exit(0);
		}

#if 0
		cudaStatus = cudaMemcpy(h_mReconInfoData.weightProj, d_mReconInfoData.weightProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Variable h_mReconInfoData.filter cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}

		for (int j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			std::cout << h_mReconInfoData.weightProj[j].x << " + " << h_mReconInfoData.weightProj[j].y << " i" << "  ";
		}
#endif

		// ͶӰ����IFFT
		cufftExecC2C(planProj, (cufftComplex*)d_mReconInfoData.weightProj, (cufftComplex*)d_mReconInfoData.weightProj, CUFFT_INVERSE);    // execute IFFT
		processFilteredProjIfft << <gridSizeWProj, blockSizeWProj >> > (mImagingSystemInfo, d_mReconInfoData);        // ���Ա任�����еĸ���

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "processFilteredProjIfftKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processFilteredProjIfftKernel!\n", cudaStatus);
			exit(0);
		}

		// ����˲����ͶӰ
		cudaStatus = cudaMemcpy(h_mReconInfoData.filterProj, d_mReconInfoData.filterProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Variable d_mReconInfoData.proj cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}

		/*for (size_t j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			std::cout << h_mReconInfoData.filterProj[j] << " ";
		}*/

		for (size_t j = 0; j < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV; j++)
		{
			h_mReconInfoData.totalProj[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * i + j] = h_mReconInfoData.filterProj[j];
		}

		// ----------------------------------��Ͷ------------------------------------
		// ��������
		cudaMemcpyToArray(cuArray, 0, 0, d_mReconInfoData.filterProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(float), cudaMemcpyDeviceToDevice);  // ��������

		angle = i * mImagingSystemInfo.thetaStep * mImagingSystemInfo.RotatedDirection;

		reconstructeImage << <gridSizeRec, blockSizeRec >> > (angle, texObj, mImagingSystemInfo, d_mReconInfoData, mGeometryPara);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reconstructeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reconstructeKernel!\n", cudaStatus);
		}

		SetConsoleCursorPosition(hConsole, coord);  //�ƶ���� 
		/*std::cout.precision(2);
		std::cout << (float)(i+1) / mImagingSystemInfo.views * 100 << "%" << std::endl;*/
		printf("%.2f%%\n", (float)(i + 1) / mImagingSystemInfo.views * 100);
	
	}
	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, g_start, g_stop);
	std::cout << "==>>�ؽ�ͼ������ʱ��(GPU)��" << elapsedTime / 1000.0f << " s" << std::endl;
	cudaEventDestroy(g_start);
	cudaEventDestroy(g_stop);

	size_t sizeImage = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ * sizeof(float);
	cudaStatus = cudaMemcpy(h_mReconInfoData.imageRecon, d_mReconInfoData.imageRecon, sizeImage, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "variable h_mReconInfoData.imageRecon cudaMemcpy failed!");
	}
	std::cout << "�ؽ���ɣ�" << std::endl;
	



	CUDAFREE(d_mReconInfoData.detU);
	CUDAFREE(d_mReconInfoData.detV);
	CUDAFREE(d_mReconInfoData.filter);
	CUDAFREE(d_mReconInfoData.filterProj);
	CUDAFREE(d_mReconInfoData.imageRecon);
	CUDAFREE(d_mReconInfoData.proj);
	CUDAFREE(d_mReconInfoData.totalProj);
	CUDAFREE(d_mReconInfoData.weightProj);
	CUDAFREE(d_mReconInfoData.x);
	CUDAFREE(d_mReconInfoData.y);
	CUDAFREE(d_mReconInfoData.z);
}


// Choose which GPU to run on, change this on a multi-GPU system.
void chooseGPU(uchar index)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(index);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}



void computeDetectorPoints(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& d_mReconInfoData)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(&d_mReconInfoData.detU, mImagingSystemInfo.dNumU * sizeof(float));
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.detU cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&d_mReconInfoData.detV, mImagingSystemInfo.dNumV * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.detV cudaMemcpy failed!");
	}

	dim3 blockSizeIPC(BLOCKSIZEY, BLOCKSIZEZ);
	dim3 gridSizeIPC((mImagingSystemInfo.dNumU - 1) / blockSizeIPC.x + 1, (mImagingSystemInfo.dNumV - 1) / blockSizeIPC.y + 1);

	// Timing
	cudaEvent_t g_start, g_stop;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	cudaEventRecord(g_start, 0);

	computeDetecPointsCoorsKernel << <gridSizeIPC, blockSizeIPC >> > (mImagingSystemInfo, d_mReconInfoData);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeIntPointCoordinatesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeIntPointCoordinatesKernel!\n", cudaStatus);
	}

	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, g_start, g_stop);

	cudaEventDestroy(g_start);
	cudaEventDestroy(g_stop);

	//totalTime += elapsedTime;

	/*cudaMemcpy(d_mReconInfoData.detV, d_mReconInfoData.detU, mImagingSystemInfo.dNumU * sizeof(float), cudaMemcpyDeviceToHost);*/
#if 0
	cudaMemcpy(d_mReconInfoData.detV, d_mReconInfoData.detV, mImagingSystemInfo.dNumV * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < mImagingSystemInfo.dNumV; i++)
	{
		std::cout << d_mReconInfoData.detV[i] << " " << std::endl;
	}

#endif
	std::cout << "����̽������Ԫ��������ʱ��(GPU)��" << elapsedTime << " ms" << std::endl;
}

void filterFT(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& h_mReconInfoData, ReconInfoData& d_mReconInfoData)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(&d_mReconInfoData.filter, mImagingSystemInfo.dNumU * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.filter cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(d_mReconInfoData.filter, h_mReconInfoData.filter, mImagingSystemInfo.dNumU * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "variable d_mReconInfoData.filter cudaMemcpy failed!");
	}
	// �˲���FFT
	cufftHandle planFilter;                           // Create cuda library function handle
	cufftPlan1d(&planFilter, mImagingSystemInfo.dNumU, CUFFT_C2C, 1);    // Plan declaration
	cufftExecC2C(planFilter, (cufftComplex*)d_mReconInfoData.filter, (cufftComplex*)d_mReconInfoData.filter, CUFFT_FORWARD);  //execute FFT
	// ȡģ
	filterAmplitude << <mImagingSystemInfo.dNumU / 1024 + 1, 1024 >> > (d_mReconInfoData.filter, mImagingSystemInfo.dNumU);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filterAmplitudeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching filterAmplitudeKernel!\n", cudaStatus);
	}

#if 0
	cudaStatus = cudaMemcpy(h_mReconInfoData.filter, d_mReconInfoData.filter, mImagingSystemInfo.dNumU * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable h_mReconInfoData.filter cudaMemcpy failed! %s.\n", cudaGetErrorString(cudaStatus));
		exit(0);
	}

	/*for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		std::cout << h_mReconInfoData.filter[i].x << " " << std::endl;
	}*/
#endif // 0
}

void prepareReconVariables(ImagingSystemInfo mImagingSystemInfo, ReconInfoData& d_mReconInfoData)
{
	cudaError_t cudaStatus;

	// �洢�ؽ�ͼ�������������
	cudaStatus = cudaMalloc(&d_mReconInfoData.x, mImagingSystemInfo.pNumX * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable imageCoordinate->x cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc(&d_mReconInfoData.y, mImagingSystemInfo.pNumY * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable imageCoordinate->y cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc(&d_mReconInfoData.z, mImagingSystemInfo.pNumZ * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable imageCoordinate->z cudaMalloc failed!");
	}

	dim3 blockSizeRec(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
	dim3 gridSizeRec((mImagingSystemInfo.pNumX - 1) / blockSizeRec.x + 1, (mImagingSystemInfo.pNumY - 1) / blockSizeRec.y + 1, (mImagingSystemInfo.pNumZ - 1) / blockSizeRec.z + 1);

	// �����ؽ�ͼ������
	computeImgCoordinates << <gridSizeRec, blockSizeRec >> > (mImagingSystemInfo, d_mReconInfoData);
	
	//cudaStreamSynchronize(0);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeImgCoordinatesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(0);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(0);
	}

	// ��ȡÿ���Ƕ�ͶӰ�������
	cudaStatus = cudaMalloc(&d_mReconInfoData.proj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.proj cudaMalloc failed!");
	}

	// ��Ȩ����ͶӰ
	cudaStatus = cudaMalloc(&d_mReconInfoData.weightProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.weightProj cudaMalloc failed!");
	}

	// �˲��������
	cudaStatus = cudaMalloc(&d_mReconInfoData.filterProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.filterProj cudaMalloc failed!");
	}

	// ��ͶӰ�������
	cudaStatus = cudaMalloc(&d_mReconInfoData.imageRecon, mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Variable d_mReconInfoData.imageRecon cudaMalloc failed!");
	}

}

// ������ά�������ݴ�����Device to Device��
// texObj -- �������cuArray -- Device�д洢���ݵ�ָ�룬data -- Դ����
void createTexture2D(cudaTextureObject_t& texObj, cudaArray_t& cuArray, float* data, size_t width, size_t height)
{
	/* ���� cudaTextureObject_t ���� */
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	//cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed to by src,
	// including paddding), we dont have any padding

	// Copy data located at address h_data in host memory to device memory
	//cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice);  // �⴮����ᵼ�º���memset()����.
	cudaMemcpyToArray(cuArray, 0, 0, data, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;   // Խ����䷽ʽ
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;                     // ����ԭʼ�������ͣ�����һ��

	// Create texture object
	//cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}

// HANDLE hConsole ���� ��Ļ�ߴ� ������COORD coord ���� ������� ����  ���x��  ���y�� 
void getCurrentCursorCoordinate(HANDLE& hConsole, COORD& coord)
{
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);  //�����Ļ�ߴ� 
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(hConsole, &csbi);
	short x = csbi.dwCursorPosition.X;
	short y = csbi.dwCursorPosition.Y;
	coord = { x,  y };
}




// ----------------------------------- Kernel function --------------------------------------------

__global__ void computeDetecPointsCoorsKernel(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData)
{
	// ������ֵ�����
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int z = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < mImagingSystemInfo.dNumU && z < mImagingSystemInfo.dNumV)
	{
		// ̽������Ԫ����
		d_mReconInfoData.detU[y] = -mImagingSystemInfo.dHalfLU + mImagingSystemInfo.dSize / 2 + y * mImagingSystemInfo.dSize;
		d_mReconInfoData.detV[z] = -mImagingSystemInfo.dHalfLV + mImagingSystemInfo.dSize / 2 + z * mImagingSystemInfo.dSize;
	}
}

__global__ void filterAmplitude(cufftComplex* d_filter, size_t dNum)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < dNum)
	{
		d_filter[x].x = sqrtf(powf(d_filter[x].x, 2) + powf(d_filter[x].y, 2));  // ��ʵ���洢ģ
	}
}

__global__ void computeImgCoordinates(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < mImagingSystemInfo.pNumX) //&& y < mImagingSystemInfo.pNumY && z < mImagingSystemInfo.pNumZ)
	{
		// �˴�������������ϵΪ��׼�� �����ؽ�ͼ�����������
		d_mReconInfoData.x[x] = -mImagingSystemInfo.imgReconLenX / 2 + mImagingSystemInfo.pSizeX / 2.0 + x * mImagingSystemInfo.pSizeX;
		d_mReconInfoData.y[y] = -mImagingSystemInfo.imgReconLenY / 2 + mImagingSystemInfo.pSizeY / 2.0 + y * mImagingSystemInfo.pSizeY;
		d_mReconInfoData.z[z] = -mImagingSystemInfo.imgReconLenZ / 2 + mImagingSystemInfo.pSizeZ / 2.0 + z * mImagingSystemInfo.pSizeZ;
	}
}

__global__ void weightProjection(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData)
{
	size_t y = blockIdx.x * blockDim.x + threadIdx.x;
	size_t z = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < mImagingSystemInfo.dNumU && z < mImagingSystemInfo.dNumV)
	{
		d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].x = d_mReconInfoData.proj[z * mImagingSystemInfo.dNumU + y] * mImagingSystemInfo.sdd / sqrtf(powf(d_mReconInfoData.detU[y], 2) + powf(d_mReconInfoData.detV[z], 2) + powf(mImagingSystemInfo.sdd, 2));
		d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].y = 0;
	}
}

__global__ void projectionFilterInTDF(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData)
{
	size_t y = threadIdx.x + blockIdx.x * blockDim.x;
	size_t z = threadIdx.y + blockIdx.y * blockDim.y;
	if (y < mImagingSystemInfo.dNumU && z < mImagingSystemInfo.dNumV)
	{
		d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].x = d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].x * d_mReconInfoData.filter[y].x;
		d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].y = d_mReconInfoData.weightProj[z * mImagingSystemInfo.dNumU + y].y * d_mReconInfoData.filter[y].x;
	}
}

__global__ void processFilteredProjIfft(ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < mImagingSystemInfo.dNumU && y < mImagingSystemInfo.dNumV)
	{
		d_mReconInfoData.filterProj[y * mImagingSystemInfo.dNumU + x] = d_mReconInfoData.weightProj[y * mImagingSystemInfo.dNumU + x].x / mImagingSystemInfo.dNumU;
	}
}

__global__ void reconstructeImage(float angle, cudaTextureObject_t texProj, ImagingSystemInfo mImagingSystemInfo, ReconInfoData d_mReconInfoData, GeometryPara mGeometryPara)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < mImagingSystemInfo.pNumX && y < mImagingSystemInfo.pNumY && z < mImagingSystemInfo.pNumZ)
	{
		angle = angle / 180 * PI;
		float tImgX = d_mReconInfoData.x[x] * cosf(angle) - d_mReconInfoData.y[y] * sinf(angle);
		float tImgY = d_mReconInfoData.y[x] * sinf(angle) + d_mReconInfoData.y[y] * cosf(angle);

		// ���ص������Ӧ��̽�����ϵ�����
		float u = mImagingSystemInfo.sdd * tImgY / (mImagingSystemInfo.sod + tImgX); //+ dHalfY) / dSize;
		float v = mImagingSystemInfo.sdd * d_mReconInfoData.z[z] / (mImagingSystemInfo.sod + tImgX); //+ dHalfZ) / dSize;
		
		// �˴��������̽����������������ϵΪ��׼�ģ�Ҳ��Ϊ�����嶥�����϶ˡ� ��ʱ��תУ��ʱ�����Ƕ�����ʱ�롣
		float correctedU = (u * cosf(mGeometryPara.beta) - v * sinf(mGeometryPara.beta) + mImagingSystemInfo.dHalfLU) / mImagingSystemInfo.dSize + mGeometryPara.offSetDetecW;    // offsetWָ̽������U(X)����
		float correctedV = (u * sinf(mGeometryPara.beta) + v * cosf(mGeometryPara.beta) + mImagingSystemInfo.dHalfLV) / mImagingSystemInfo.dSize + mGeometryPara.offSetDetecH;

		//imgRec[z * width * height + y * width + x] = v / pSizeY;

		//// Read from texture and write to global memory
		d_mReconInfoData.imageRecon[z * mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY + y * mImagingSystemInfo.pNumX + x] += (mImagingSystemInfo.sod * mImagingSystemInfo.sod) / ((mImagingSystemInfo.sod + tImgX) * (mImagingSystemInfo.sod + tImgX)) * tex2D<float>(texProj, correctedU, correctedV) * mImagingSystemInfo.thetaStep;// / mImagingSystemInfo.imgReconLenX; // (mImagingSystemInfo.pNumX * mImagingSystemInfo.pSizeX);
		//d_mReconInfoData.imageRecon[z * mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY + y * mImagingSystemInfo.pNumX + x] += 1 / 2.0f * (mImagingSystemInfo.sod * mImagingSystemInfo.sod) / ((mImagingSystemInfo.sod + tImgX) * (mImagingSystemInfo.sod + tImgX)) * tex2D<float>(texProj, correctedU + 0.5, correctedV + 0.5) * mImagingSystemInfo.thetaStep;// / mImagingSystemInfo.imgReconLenX; // (mImagingSystemInfo.pNumX * mImagingSystemInfo.pSizeX);
		
																																																																																					   // ���������ֵʱ���Ƿ���Ҫ��0.5   ????		
		/*imgRec[z * width * height + y * width + x] += ((sod * sdd) / pow((sod + imgX[x] * sinf(angle) - imgY[y] * cosf(angle)), 2))
			* tex3D<float>(texProj, u, v, num + 0.5) / width;*/
			// u, vӦ�ó���dSize��ĿǰdSize��1, ������Ҫ���Գ���dSize
	}
}
 