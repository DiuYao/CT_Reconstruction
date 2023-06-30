/***********************FDK_CUDA*************************
* 几何标准: 左手坐标系
* 旋转中心为原点，射线方向为X轴正方向。
* 探测器: 横向为U轴, 向右为正方向; 纵向为V轴, 向上为正方向
* Version  1.1
* CUDA Version: 11.8
* Author: Du
**********************************************************/

/************************ Log ****************************
* 2023年5月14日 
* 1.加入重建图像实际大小 单位 mm
* 2.增加重建信息输出内容，包括图像大小，图像像素大小
**********************************************************/

#pragma once

#include <xstring>

#include "Kernel.cuh"

#define DELETENEW(arrP)\
if (arrP != nullptr){\
delete[] arrP;\
arrP = nullptr;\
}



// 坐标：图像坐标
//struct Coordinate3D
//{
//	float* x, * y, * z;
//};

struct ImagingSystemInfo
{
	size_t pNumX, pNumY, pNumZ;   // 图像像素数
	float imgReconLenX, imgReconLenY, imgReconLenZ;   // mm
	size_t sod, sdd;
	size_t views;
	size_t dNumU, dNumV;
	size_t dNumUPaddingZero;

	float dSize;    // 探测器分辨率


	float totalTime = 0.0f;

	// Compute parameters
	float thetaStep;
	float dHalfLU;
	float dHalfLV;
	float horizontalR;    // 水平视野圆半径
	float verticalR;      // 垂直视野圆半径

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
	size_t intNum;        // 某条射线上的积分点个数

	int RotatedDirection;  // 扫描时的旋转方向
};

struct GeometryPara
{
	float beta;        // 探测器倾角, 单位 度
	float offSetDetecW;   // 探测器X方向偏移, 单位 pixel, 左负右正
	float offSetDetecH;   // 探测器Y方向偏移, 单位 pixel, 上负下正
	// note: 方向和正负关系有待考证
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
	// GPU指针变量
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

