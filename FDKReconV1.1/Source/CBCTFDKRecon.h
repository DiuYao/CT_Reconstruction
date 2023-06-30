/***********************FDK_CUDA*************************
* ���α�׼: ��������ϵ
* ��ת����Ϊԭ�㣬���߷���ΪX��������
* ̽����: ����ΪU��, ����Ϊ������; ����ΪV��, ����Ϊ������
* Version  1.1
* CUDA Version: 11.8
* Author: Du
**********************************************************/

/************************ Log ****************************
* 2023��5��14�� 
* 1.�����ؽ�ͼ��ʵ�ʴ�С ��λ mm
* 2.�����ؽ���Ϣ������ݣ�����ͼ���С��ͼ�����ش�С
**********************************************************/

#pragma once

#include <xstring>

#include "Kernel.cuh"

#define DELETENEW(arrP)\
if (arrP != nullptr){\
delete[] arrP;\
arrP = nullptr;\
}



// ���꣺ͼ������
//struct Coordinate3D
//{
//	float* x, * y, * z;
//};

struct ImagingSystemInfo
{
	size_t pNumX, pNumY, pNumZ;   // ͼ��������
	float imgReconLenX, imgReconLenY, imgReconLenZ;   // mm
	size_t sod, sdd;
	size_t views;
	size_t dNumU, dNumV;
	size_t dNumUPaddingZero;

	float dSize;    // ̽�����ֱ���


	float totalTime = 0.0f;

	// Compute parameters
	float thetaStep;
	float dHalfLU;
	float dHalfLV;
	float horizontalR;    // ˮƽ��ҰԲ�뾶
	float verticalR;      // ��ֱ��ҰԲ�뾶

	/*float horizontalR = 289 / 2;
	float verticalR = 180 / 2;*/

	// Pixel size
	float pSizeX;			// mm
	float pSizeY;
	float pSizeZ;
	// Integral calculus
	float dx;
	float dy;
	float dz;
	// Compute detector coordinates
	size_t intNum;        // ĳ�������ϵĻ��ֵ����

	int RotatedDirection;  // ɨ��ʱ����ת����
};

struct GeometryPara
{
	float beta;        // ̽�������, ��λ ��
	float offSetDetecW;   // ̽����X����ƫ��, ��λ pixel, ������
	float offSetDetecH;   // ̽����Y����ƫ��, ��λ pixel, �ϸ�����
	// note: �����������ϵ�д���֤
};

struct ReconInfoData
{
	float* detU;
	float* detV;

#if _DEBUG
	float* detTempU;
	float* detTempV;
	
#endif
	
	cufftComplex* filter;
	float* totalProj;
	float* proj;
	cufftComplex* weightProj;
	float* filterProj;

	float* imageRecon;
	
	float* x, * y, * z;
};

struct FilePath
{
	
	std::string readProjPath;
	std::string readAnglePath;
	std::string writeReconPath;
	std::string writeFilterProjPath;
};

class CBCTFDKRecon
{
private:
	ImagingSystemInfo mImagingSystemInfo;
	GeometryPara mGeometryPara;
	FilePath mFilrePath;

private:
	
	ReconInfoData h_mReconInfoData = { nullptr };
	// GPUָ�����
	ReconInfoData d_mReconInfoData = { nullptr };

public:
	CBCTFDKRecon();
	~CBCTFDKRecon();

	void recon();

private:
	void getReconConfig();
	void readProj();
	void saveAsImage();
	void computeParas();

	void designFilter();

	void negativeValueToZero();
	void normalizeData();
	void normalizeDataOptimize();

	void findMaxMinValue(float* max, float* min);

	void printReconInfo();
};

