#include "CBCTFDKRecon.h"
#include "InitFile.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;


CBCTFDKRecon::CBCTFDKRecon()
{
	getReconConfig();

	h_mReconInfoData.imageRecon = new float[mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ];
	h_mReconInfoData.x = new float[mImagingSystemInfo.pNumX];
	h_mReconInfoData.y = new float[mImagingSystemInfo.pNumY];
	h_mReconInfoData.z = new float[mImagingSystemInfo.pNumZ];
	//h_mReconInfoData.proj = new float[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
	//h_mReconInfoData.weightProj = new cufftComplex[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
	h_mReconInfoData.filterProj = new float[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
	h_mReconInfoData.totalProj = new float[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views];

}

CBCTFDKRecon::~CBCTFDKRecon()
{
	DELETENEW(h_mReconInfoData.detU);
	DELETENEW(h_mReconInfoData.detV);
	DELETENEW(h_mReconInfoData.filter);
	DELETENEW(h_mReconInfoData.filterProj);
	DELETENEW(h_mReconInfoData.imageRecon);
	DELETENEW(h_mReconInfoData.proj);
	DELETENEW(h_mReconInfoData.totalProj);
	DELETENEW(h_mReconInfoData.weightProj);
	DELETENEW(h_mReconInfoData.x);
	DELETENEW(h_mReconInfoData.y);
	DELETENEW(h_mReconInfoData.z);
#if _DEBUG
	DELETENEW(h_mReconInfoData.detTempU);
	DELETENEW(h_mReconInfoData.detTempV);
#endif
}

void CBCTFDKRecon::recon()
{
	//chooseGPU(0);

	computeParas();
	printReconInfo();

	readProj();

	designFilter();

#if 0
	for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		cout << h_mReconInfoData.filter[i].x << endl;
	}
#endif
	
	reconGPU(mImagingSystemInfo, h_mReconInfoData, d_mReconInfoData, mGeometryPara);

	std::cout << "处理重建图像==>>" << std::endl;
	// 负值置0
	if (mImagingSystemInfo.reconZeroFlag == 1)
	{
		negativeValueToZero();
	}
	
	// 负值置0，并归一化
	//normalizeDataOptimize();
	cout << "处理重建图像结束" << endl;

	saveAsImage();
	cout << endl << "图像存储完成" << endl;

	saveReconInfo();

#if 0
	for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		cout << d_mReconInfoData.detTempU[i] << " " << endl;
	}

	cout << "-----------------------------------------" << endl;

	for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		cout << d_mReconInfoData.detTempV[i] << " " << endl;
	}
#endif
}



void CBCTFDKRecon::getReconConfig()
{
	int ModeNum = 1;

	//string Workdir;
	CInitFile tmpInitFile;

	string modename = "Config/Reconstruct.ini";
	tmpInitFile.GetFileName(modename.c_str());


	char section[32];
	sprintf(section, "Mode%d", 1);

	tmpInitFile.GetEntryValue(section, "PNumX", -100, mImagingSystemInfo.pNumX);
	tmpInitFile.GetEntryValue(section, "PNumY", -100, mImagingSystemInfo.pNumY);
	tmpInitFile.GetEntryValue(section, "PNumZ", -100, mImagingSystemInfo.pNumZ);

	if (mImagingSystemInfo.pNumX < 0 || mImagingSystemInfo.pNumY < 0 || mImagingSystemInfo.pNumZ < 0)
	{
		cout << "图像大小(pix)读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "ImgReconLenX", -100, mImagingSystemInfo.imgReconLenX);
	tmpInitFile.GetEntryValue(section, "ImgReconLenY", -100, mImagingSystemInfo.imgReconLenY);
	tmpInitFile.GetEntryValue(section, "ImgReconLenY", -100, mImagingSystemInfo.imgReconLenZ);

	if (mImagingSystemInfo.imgReconLenX < 0 || mImagingSystemInfo.imgReconLenY < 0 || mImagingSystemInfo.imgReconLenZ < 0)
	{
		cout << "图像大小(mm)读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "RotatedDerection", 0, mImagingSystemInfo.RotatedDirection);
	if (mImagingSystemInfo.RotatedDirection == 0)
	{
		cout << "未获得旋转方向！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "SOD", -100, mImagingSystemInfo.sod);
	tmpInitFile.GetEntryValue(section, "SDD", -100, mImagingSystemInfo.sdd);
	tmpInitFile.GetEntryValue(section, "Views", -100, mImagingSystemInfo.views);

	if (mImagingSystemInfo.sod < 0 || mImagingSystemInfo.sdd < 0 || mImagingSystemInfo.views < 0)
	{
		cout << "sod or sdd or views读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "DNumU", -100, mImagingSystemInfo.dNumU);
	tmpInitFile.GetEntryValue(section, "DNumV", -100, mImagingSystemInfo.dNumV);
	tmpInitFile.GetEntryValue(section, "DSize", -100, mImagingSystemInfo.dSize);

	if (mImagingSystemInfo.dNumU < 0 || mImagingSystemInfo.dNumV < 0 || mImagingSystemInfo.dSize < 0)
	{
		cout << "探测器参数读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "OffsetW", -100, mGeometryPara.offSetDetecW);
	tmpInitFile.GetEntryValue(section, "OffsetH", -100, mGeometryPara.offSetDetecH);
	tmpInitFile.GetEntryValue(section, "Beta", -100, mGeometryPara.beta);
	mGeometryPara.beta = mGeometryPara.beta / 180 * PI;  // 换位弧度，后续使用是弧度
	if (mImagingSystemInfo.dNumU < 0 || mImagingSystemInfo.dNumV < 0 || mImagingSystemInfo.dSize < 0)
	{
		cout << "几何参数读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}



	tmpInitFile.GetEntryValue(section, "CTDataFolder", "null", mFilrePath.CTDataFolder);
	tmpInitFile.GetEntryValue(section, "ProjName", "null", mFilrePath.projName);
	tmpInitFile.GetEntryValue(section, "ReadAnglePath", "null", mFilrePath.readAnglePath);

	if (mFilrePath.CTDataFolder == "null" || mFilrePath.projName == "null" || mFilrePath.readAnglePath == "null")
	{
		cout << "路径读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "ReconName", "null", mFilrePath.reconName);
	//tmpInitFile.GetEntryValue(section, "WriteFilterProjPath", "null", mFilrePath.writeFilterProjPath);
	if (mFilrePath.reconName == "null") //|| mFilrePath.writeFilterProjPath == "null")
	{
		cout << "输出路径读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	// 读取 投影负值置零标识和重建图像负值置零标识
	tmpInitFile.GetEntryValue(section, "IsProjZero", -1, mImagingSystemInfo.projZeroFlag);
	tmpInitFile.GetEntryValue(section, "IsReconZero", -1, mImagingSystemInfo.reconZeroFlag);

	if (mImagingSystemInfo.projZeroFlag == -1 || mImagingSystemInfo.reconZeroFlag == -1)
	{
		cout << "投影或重建置零标识读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

	// 读取 投影滤波后是否保存的标识
	tmpInitFile.GetEntryValue(section, "IsFilteredPorjSaving", -1, mImagingSystemInfo.filteredPorjSaveFalg);

	if (mImagingSystemInfo.filteredPorjSaveFalg == -1)
	{
		cout << "滤波后的投影的保存标识读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}


	// 读取使用的GPU序号
	tmpInitFile.GetEntryValue(section, "GPUIndex", -1, mImagingSystemInfo.deviceIndex);
	
	if (mImagingSystemInfo.deviceIndex == -1)
	{
		cout << "GPU序号读取有误！" << endl;
		system("pause");
		exit(0);
		return;
	}

#if _DEBUG
	cout << "pNumX = " << mImagingSystemInfo.pNumX << endl;
	cout << "pNumY = " << mImagingSystemInfo.pNumY << endl;
	cout << "pNumZ = " << mImagingSystemInfo.pNumZ << endl;
	cout << "sod = " << mImagingSystemInfo.sod << endl;
	cout << "sdd = " << mImagingSystemInfo.sdd << endl;
	cout << "dNumY = " << mImagingSystemInfo.dNumU << endl;
	cout << "dNumZ = " << mImagingSystemInfo.dNumV << endl;
	cout << "offsetW = " << mGeometryPara.offSetDetecW << endl;
	cout << "offsetH = " << mGeometryPara.offSetDetecH << endl;
	cout << "beta = " << mGeometryPara.beta << endl;

	cout << "projZeroFlag = " << mImagingSystemInfo.projZeroFlag << endl;
	cout << "reconZeroFlag = " << mImagingSystemInfo.reconZeroFlag << endl;
	cout << "deviceIndex = " << mImagingSystemInfo.deviceIndex << endl;

	cout << "readProjPath = " << mFilrePath.CTDataFolder << endl;
	cout << "readAnglePath = " << mFilrePath.readAnglePath << endl;
	cout << "reconName = " << mFilrePath.reconName << endl;
	cout << "writeFilterProjPath = " << mFilrePath.writeFilterProjPath << endl;

	printf("Find %d mode->Select Mode1\n", ModeNum);

#endif // INI
}

void CBCTFDKRecon::saveReconInfo()
{
	

	//string Workdir;
	CInitFile tmpInitFile;

	string modename = mFilrePath.reconFolder + "\\Reconstruct.ini";

	if (_access(modename.c_str(), 0) == 0)
	{
		remove(modename.c_str());
	}

	tmpInitFile.SaveFile(modename.c_str());
	tmpInitFile.GetFileName(modename.c_str());  // 存在问题：保存结果第一行多了[]
	

	char section[32] = "Recon";

	tmpInitFile.SetEntryValue(section, "RotatedDerection", mImagingSystemInfo.RotatedDirection);
	// 读取 投影负值置零标识和重建图像负值置零标识
	tmpInitFile.SetEntryValue(section, "IsProjZero", mImagingSystemInfo.projZeroFlag);
	tmpInitFile.SetEntryValue(section, "IsReconZero", mImagingSystemInfo.reconZeroFlag);

	tmpInitFile.SetEntryValue(section, "PNumX", mImagingSystemInfo.pNumX);
	tmpInitFile.SetEntryValue(section, "PNumY", mImagingSystemInfo.pNumY);
	tmpInitFile.SetEntryValue(section, "PNumZ", mImagingSystemInfo.pNumZ);

	tmpInitFile.SetEntryValue(section, "ImgReconLenX", mImagingSystemInfo.imgReconLenX);
	tmpInitFile.SetEntryValue(section, "ImgReconLenY", mImagingSystemInfo.imgReconLenY);
	tmpInitFile.SetEntryValue(section, "ImgReconLenY", mImagingSystemInfo.imgReconLenZ);


	tmpInitFile.SetEntryValue(section, "SOD", mImagingSystemInfo.sod);
	tmpInitFile.SetEntryValue(section, "SDD", mImagingSystemInfo.sdd);
	tmpInitFile.SetEntryValue(section, "Views", mImagingSystemInfo.views);

	tmpInitFile.SetEntryValue(section, "DNumU", mImagingSystemInfo.dNumU);
	tmpInitFile.SetEntryValue(section, "DNumV", mImagingSystemInfo.dNumV);
	tmpInitFile.SetEntryValue(section, "DSize", mImagingSystemInfo.dSize);

	tmpInitFile.SetEntryValue(section, "OffsetW", mGeometryPara.offSetDetecW);
	tmpInitFile.SetEntryValue(section, "OffsetH", mGeometryPara.offSetDetecH);

	mGeometryPara.beta = mGeometryPara.beta / PI * 180;  // 换为角度制
	tmpInitFile.SetEntryValue(section, "Beta", mGeometryPara.beta);


	tmpInitFile.SetEntryValue(section, "CTDataFolder", mFilrePath.CTDataFolder);
	tmpInitFile.SetEntryValue(section, "ProjName", mFilrePath.projName);

	mFilrePath.reconName += "_" + to_string(mImagingSystemInfo.pNumX) + 'x' + to_string(mImagingSystemInfo.pNumY) + 'x' + to_string(mImagingSystemInfo.pNumZ) + "_float.raw";
	tmpInitFile.SetEntryValue(section, "ReconName", mFilrePath.reconName);

	// 读取 投影滤波后是否保存的标识
	tmpInitFile.SetEntryValue(section, "IsFilteredPorjSaving", mImagingSystemInfo.filteredPorjSaveFalg);

}

void CBCTFDKRecon::readProj()
{
	string projPath = mFilrePath.CTDataFolder + "/" + mFilrePath.projName;

	std::cout << "正在读取投影数据==>>" << std::endl;
	
	ifstream ifs;
	ifs.open(projPath, ios::in | ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "文件打开失败！" << std::endl;
		system("pause");
		exit(0);
	}
	ifs.read((char*)h_mReconInfoData.totalProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views * sizeof(float));

	std::cout << "投影数据读取完成！" << std::endl;

	// 投影图中负值置0
	if (mImagingSystemInfo.projZeroFlag == 1)
	{
		for (size_t i = 0; i < mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views; ++i)
		{
			if (h_mReconInfoData.totalProj[i] < 0)
			{
				h_mReconInfoData.totalProj[i] = 0;
			}
		}
	}
	

}

void CBCTFDKRecon::saveAsImage()
{
	mFilrePath.reconFolder = mFilrePath.CTDataFolder + "\\Recon\\" + mFilrePath.reconName;

	if (_access(mFilrePath.reconFolder.c_str(), 0) == -1)
	{
		// 删除非空文件夹未实现成功，暂且使用删除文件
		//string command = "rd " + mFilrePath.reconFolder;
		//system(command.c_str());
		//remove(mFilrePath.reconFolder.c_str());

		// 创建文件夹
		string command = "mkdir " + mFilrePath.reconFolder;
		system(command.c_str());
	}

	// 此处用"\\"是为了和读取配置文件自行处理后中使用"\\"保持一致，使用"/"也可
	string reconPath = mFilrePath.reconFolder + "\\" + mFilrePath.reconName + "_" + to_string(mImagingSystemInfo.pNumX) + 'x' + to_string(mImagingSystemInfo.pNumY) + 'x' + to_string(mImagingSystemInfo.pNumZ) + "_float.raw";

	if (_access(reconPath.c_str(), 0) == 0)
	{
		remove(reconPath.c_str());
	}
	

	// 保存
	std::cout << "正在保存重建图像==>>" << std::endl;

	ofstream ofs;
	ofs.open(reconPath, ios::out | ios::binary);
	ofs.write((const char*)h_mReconInfoData.imageRecon, mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ * sizeof(float));
	ofs.close();

	std::cout << "重建图像保存完成" << std::endl;


	// 滤波后的投影保存
	
	if (mImagingSystemInfo.filteredPorjSaveFalg)
	{
		string filterProjFolder = mFilrePath.CTDataFolder + "\\FilteredProj";

		if (_access(filterProjFolder.c_str(), 0) == -1)
		{
			// 创建文件夹
			string command = "mkdir " + filterProjFolder;
			system(command.c_str());
		}
		
		string filterProjPath = filterProjFolder + "/" + "FilteredProj_" + to_string(mImagingSystemInfo.dNumU) + 'x' + to_string(mImagingSystemInfo.dNumV) + 'x' + to_string(mImagingSystemInfo.views) + "_float.raw";

		if (_access(reconPath.c_str(), 0) == 0)
		{
			remove(filterProjPath.c_str());
		}

		cout << endl << "正在保存滤波后的投影==>>" << endl;

		ofs.open(filterProjPath, ios::out | ios::binary);
		ofs.write((const char*)h_mReconInfoData.totalProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views * sizeof(float));
		ofs.close();

		std::cout << "滤波后的投影保存完成" << std::endl;
	}
	else
	{
		cout << endl << "不保存滤波后的投影" << endl;
	}
}


void CBCTFDKRecon::computeParas()
{
	mImagingSystemInfo.totalTime = 0.0f;

	// Compute parameters
	if (mFilrePath.readAnglePath == "")
	{
		mImagingSystemInfo.thetaStep = (float)360 / mImagingSystemInfo.views;
	}
	else
	{
		
	}
	
	mImagingSystemInfo.dHalfLU = mImagingSystemInfo.dNumU * mImagingSystemInfo.dSize / 2;
	mImagingSystemInfo.dHalfLV = mImagingSystemInfo.dNumV * mImagingSystemInfo.dSize / 2;
	mImagingSystemInfo.horizontalR = mImagingSystemInfo.sod * mImagingSystemInfo.dHalfLU / sqrt(mImagingSystemInfo.dHalfLU * mImagingSystemInfo.dHalfLU + mImagingSystemInfo.sdd * mImagingSystemInfo.sdd);     // 水平视野圆半径
	mImagingSystemInfo.verticalR = mImagingSystemInfo.sod * mImagingSystemInfo.dHalfLV / sqrt(mImagingSystemInfo.dHalfLV * mImagingSystemInfo.dHalfLV + mImagingSystemInfo.sdd * mImagingSystemInfo.sdd);       // 垂直视野圆半径

	/*float horizontalR = 289 / 2;
	float verticalR = 180 / 2;*/

	// Pixel size
	/*mImagingSystemInfo.pSizeX = 2 * mImagingSystemInfo.horizontalR / mImagingSystemInfo.pNumX;
	mImagingSystemInfo.pSizeY = 2 * mImagingSystemInfo.horizontalR / mImagingSystemInfo.pNumY;
	mImagingSystemInfo.pSizeZ = 2 * mImagingSystemInfo.verticalR / mImagingSystemInfo.pNumZ;*/

	mImagingSystemInfo.pSizeX = mImagingSystemInfo.imgReconLenX / mImagingSystemInfo.pNumX;
	mImagingSystemInfo.pSizeY = mImagingSystemInfo.imgReconLenY / mImagingSystemInfo.pNumY;
	mImagingSystemInfo.pSizeZ = mImagingSystemInfo.imgReconLenZ / mImagingSystemInfo.pNumZ;

	// Integral calculus
	mImagingSystemInfo.dx = 0.5 * mImagingSystemInfo.pSizeX;
	mImagingSystemInfo.dy = 0.5 * mImagingSystemInfo.pSizeY;
	mImagingSystemInfo.dz = 0.5 * mImagingSystemInfo.pSizeZ;

	// FFT前补零后的数据长度
	mImagingSystemInfo.dNumUPaddingZero = powf(2, ceilf(log2f(mImagingSystemInfo.dNumU)));

	// Compute detector coordinates
	mImagingSystemInfo.intNum = roundf(2 * mImagingSystemInfo.horizontalR / mImagingSystemInfo.dx);        // 某条射线上的积分点个数,X方向
}

// 设计滤波器
void CBCTFDKRecon::designFilter()
{
	//h_mReconInfoData.filter = new cufftComplex[mImagingSystemInfo.dNumU]();
	h_mReconInfoData.filter = new cufftComplex[mImagingSystemInfo.dNumUPaddingZero]();   // 初始化为全零

	int dHalf = mImagingSystemInfo.dNumU / 2;
	for (int i = 0; i < mImagingSystemInfo.dNumU; i++)
	{
		if ((i - dHalf) % 2 == 0)
		{
			h_mReconInfoData.filter[i].x = 0;
		}
		else
		{
			//h_mReconInfoData.filter[i].x = -1 / (PI * PI * (i - dHalf) * mImagingSystemInfo.dSize * (i - dHalf) * mImagingSystemInfo.dSize);
			h_mReconInfoData.filter[i].x = -1 / (PI * PI * (i - dHalf) * (i - dHalf));

		}
	}
	//h_mReconInfoData.filter[dHalf].x = (float)1 / (4 * mImagingSystemInfo.dSize * mImagingSystemInfo.dSize);
	h_mReconInfoData.filter[dHalf].x = (float)1 / 4;


}

void CBCTFDKRecon::negativeValueToZero()
{
	// 负值置零
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (h_mReconInfoData.imageRecon[i] < 0)
		{
			h_mReconInfoData.imageRecon[i] = 0;
		}
	}

}

// 归一化 —— 利用最值
void CBCTFDKRecon::normalizeData()
{
	float max = h_mReconInfoData.imageRecon[0], min = h_mReconInfoData.imageRecon[0];

	// 找最大值
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (max < h_mReconInfoData.imageRecon[i])
		{
			max = h_mReconInfoData.imageRecon[i];
		}
	}

	// 找最小值
	for (size_t i = 0; i < elementNums; i++)
	{
		if (min > h_mReconInfoData.imageRecon[i])
		{
			min = h_mReconInfoData.imageRecon[i];
		}
	}

	// 归一化
	for (size_t i = 0; i < elementNums; i++)
	{
		//image[i] = (image[i] - min) * (end - start) / (max - min) + start;
		h_mReconInfoData.imageRecon[i] = (h_mReconInfoData.imageRecon[i] - min) / (max - min);  // 区间[0, 1].
	}
}

// 先将负值置0, 然后进行归一化 —— 利用0和最大值
void CBCTFDKRecon::normalizeDataOptimize()
{
	float max = h_mReconInfoData.imageRecon[0], min = h_mReconInfoData.imageRecon[0];
	// 找最大值
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (max < h_mReconInfoData.imageRecon[i])
		{
			max = h_mReconInfoData.imageRecon[i];
		}
	}
	// 负值置零
	for (size_t i = 0; i < elementNums; i++)
	{
		if (h_mReconInfoData.imageRecon[i] <  0)
		{
			h_mReconInfoData.imageRecon[i] = 0;
		}
	}

	// 归一化
	for (size_t i = 0; i < elementNums; i++)
	{
		//image[i] = (image[i] - min) * (end - start) / (max - min) + start;
		//h_mReconInfoData.imageRecon[i] = (h_mReconInfoData.imageRecon[i] - 0) / (max - 0);  // 区间[0, 1].
		h_mReconInfoData.imageRecon[i] = h_mReconInfoData.imageRecon[i] / max;  // 区间[0, 1].
	}
}

void CBCTFDKRecon::findMaxMinValue(float* max, float* min)
{
	

}

void CBCTFDKRecon::printReconInfo()
{
	cout << "========== 重建系统信息 >>>>>>>>>" << endl;
	cout << "sdd：" << mImagingSystemInfo.sdd << "  " << "sod：" << mImagingSystemInfo.sod << "  " << "探测器单元大小：" << mImagingSystemInfo.dSize << endl;
	cout << "重建图像大小(pix)：" << mImagingSystemInfo.pNumX << "x" << mImagingSystemInfo.pNumY << "x" << mImagingSystemInfo.pNumZ << endl;
	cout << "重建图像大小(mm)：" << mImagingSystemInfo.imgReconLenX << "x" << mImagingSystemInfo.imgReconLenY << "x" << mImagingSystemInfo.imgReconLenZ << endl;
	cout << "重建图像像素大小(mm)：" << mImagingSystemInfo.pSizeX << "x" << mImagingSystemInfo.pSizeY << "x" << mImagingSystemInfo.pSizeZ << endl;
	cout << "一周采集投影个数：" << mImagingSystemInfo.views << endl;
	cout << "投影尺寸：" << mImagingSystemInfo.dNumU << "x" << mImagingSystemInfo.dNumV << "x" << mImagingSystemInfo.views << endl << endl;
}







