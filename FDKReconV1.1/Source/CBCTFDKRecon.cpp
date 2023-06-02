#include "CBCTFDKRecon.h"
#include "InitFile.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;


CBCTFDKRecon::CBCTFDKRecon()
{
	getReconConfig();
	h_mReconInfoData.imageRecon = new float[mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ];
	h_mReconInfoData.x = new float[mImagingSystemInfo.pNumX];
	h_mReconInfoData.y = new float[mImagingSystemInfo.pNumY];
	h_mReconInfoData.z = new float[mImagingSystemInfo.pNumZ];
	h_mReconInfoData.proj = new float[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
	h_mReconInfoData.weightProj = new cufftComplex[mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV];
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

	std::cout << "�����ؽ�ͼ��..." << std::endl;
	// ��ֵ��0
	//negativeValueToZero();
	// ��ֵ��0������һ��
	normalizeDataOptimize();
	cout << "�����ؽ�ͼ�����" << endl;

	saveAsImage();
	cout << "ͼ��洢���" << endl;

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
		cout << "ͼ���С(pix)��ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "ImgReconLenX", -100, mImagingSystemInfo.imgReconLenX);
	tmpInitFile.GetEntryValue(section, "ImgReconLenY", -100, mImagingSystemInfo.imgReconLenY);
	tmpInitFile.GetEntryValue(section, "ImgReconLenY", -100, mImagingSystemInfo.imgReconLenZ);

	if (mImagingSystemInfo.imgReconLenX < 0 || mImagingSystemInfo.imgReconLenY < 0 || mImagingSystemInfo.imgReconLenZ < 0)
	{
		cout << "ͼ���С(mm)��ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "RotatedDerection", 0, mImagingSystemInfo.RotatedDirection);
	if (mImagingSystemInfo.RotatedDirection == 0)
	{
		cout << "δ�����ת����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "SOD", -100, mImagingSystemInfo.sod);
	tmpInitFile.GetEntryValue(section, "SDD", -100, mImagingSystemInfo.sdd);
	tmpInitFile.GetEntryValue(section, "Views", -100, mImagingSystemInfo.views);

	if (mImagingSystemInfo.sod < 0 || mImagingSystemInfo.sdd < 0 || mImagingSystemInfo.views < 0)
	{
		cout << "sod or sdd or views��ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "DNumU", -100, mImagingSystemInfo.dNumU);
	tmpInitFile.GetEntryValue(section, "DNumV", -100, mImagingSystemInfo.dNumV);
	tmpInitFile.GetEntryValue(section, "DSize", -100, mImagingSystemInfo.dSize);

	if (mImagingSystemInfo.dNumU < 0 || mImagingSystemInfo.dNumV < 0 || mImagingSystemInfo.dSize < 0)
	{
		cout << "̽����������ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "OffsetW", -100, mGeometryPara.offSetDetecW);
	tmpInitFile.GetEntryValue(section, "OffsetH", -100, mGeometryPara.offSetDetecH);
	tmpInitFile.GetEntryValue(section, "Beta", -100, mGeometryPara.beta);

	if (mImagingSystemInfo.dNumU < 0 || mImagingSystemInfo.dNumV < 0 || mImagingSystemInfo.dSize < 0)
	{
		cout << "���β�����ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}



	tmpInitFile.GetEntryValue(section, "ReadProjPath", "null", mFilrePath.readProjPath);
	tmpInitFile.GetEntryValue(section, "ReadAnglePath", "null", mFilrePath.readAnglePath);

	if (mFilrePath.readProjPath == "null" || mFilrePath.readAnglePath == "null")
	{
		cout << "·����ȡ����" << endl;
		system("pause");
		exit(0);
		return;
	}

	tmpInitFile.GetEntryValue(section, "WriteReconPath", "null", mFilrePath.writeReconPath);
	tmpInitFile.GetEntryValue(section, "WriteFilterProjPath", "null", mFilrePath.writeFilterProjPath);



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

	cout << "readProjPath = " << mFilrePath.readProjPath << endl;
	cout << "writeReconPath = " << mFilrePath.readAnglePath << endl;
	cout << "writeReconPath = " << mFilrePath.writeReconPath << endl;
	cout << "writeReconPath = " << mFilrePath.writeFilterProjPath << endl;

	printf("Find %d mode->Select Mode1\n", ModeNum);

#endif // INI
}

void CBCTFDKRecon::readProj()
{
	std::cout << "���ڶ�ȡͶӰ����..." << std::endl;
	
	ifstream ifs;
	ifs.open(mFilrePath.readProjPath, ios::in | ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "�ļ���ʧ�ܣ�" << std::endl;
		system("pause");
		exit(0);
	}
	ifs.read((char*)h_mReconInfoData.totalProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views * sizeof(float));

	std::cout << "ͶӰ���ݶ�ȡ��ɣ�" << std::endl;
}

void CBCTFDKRecon::saveAsImage()
{
	std::cout << "���ڱ����ؽ�ͼ��..." << std::endl;

	mFilrePath.writeReconPath += "_" + to_string(mImagingSystemInfo.pNumX) + 'x' + to_string(mImagingSystemInfo.pNumY) + 'x' + to_string(mImagingSystemInfo.pNumZ) + "_float.raw";
	ofstream ofs;
	ofs.open(mFilrePath.writeReconPath, ios::out | ios::binary);
	ofs.write((const char*)h_mReconInfoData.imageRecon, mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ * sizeof(float));
	ofs.close();

	std::cout << "�ؽ�ͼ�񱣴����!" << std::endl;


	std::cout << "�Ƿ���Ҫ�����˲����ͶӰ(0-N0, 1-Yes)��";
	int index = 0;
	cin >> index;
	if (index)
	{
		mFilrePath.writeFilterProjPath += "_" + to_string(mImagingSystemInfo.dNumU) + 'x' + to_string(mImagingSystemInfo.dNumV) + 'x' + to_string(mImagingSystemInfo.views) + "_float.raw";
		ofs.open(mFilrePath.writeFilterProjPath, ios::out | ios::binary);
		ofs.write((const char*)h_mReconInfoData.totalProj, mImagingSystemInfo.dNumU * mImagingSystemInfo.dNumV * mImagingSystemInfo.views * sizeof(float));
		ofs.close();

		std::cout << "�˲����ͶӰ������ɣ�" << std::endl;
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
	mImagingSystemInfo.horizontalR = mImagingSystemInfo.sod * mImagingSystemInfo.dHalfLU / sqrt(mImagingSystemInfo.dHalfLU * mImagingSystemInfo.dHalfLU + mImagingSystemInfo.sdd * mImagingSystemInfo.sdd);     // ˮƽ��ҰԲ�뾶
	mImagingSystemInfo.verticalR = mImagingSystemInfo.sod * mImagingSystemInfo.dHalfLV / sqrt(mImagingSystemInfo.dHalfLV * mImagingSystemInfo.dHalfLV + mImagingSystemInfo.sdd * mImagingSystemInfo.sdd);       // ��ֱ��ҰԲ�뾶

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
	// Compute detector coordinates
	mImagingSystemInfo.intNum = roundf(2 * mImagingSystemInfo.horizontalR / mImagingSystemInfo.dx);        // ĳ�������ϵĻ��ֵ����,X����
}

// ����˲���
void CBCTFDKRecon::designFilter()
{
	h_mReconInfoData.filter = new cufftComplex[mImagingSystemInfo.dNumU]();

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
	// ��ֵ����
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (h_mReconInfoData.imageRecon[i] < 0)
		{
			h_mReconInfoData.imageRecon[i] = 0;
		}
	}

}

// ��һ�� ���� ������ֵ
void CBCTFDKRecon::normalizeData()
{
	float max = h_mReconInfoData.imageRecon[0], min = h_mReconInfoData.imageRecon[0];

	// �����ֵ
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (max < h_mReconInfoData.imageRecon[i])
		{
			max = h_mReconInfoData.imageRecon[i];
		}
	}

	// ����Сֵ
	for (size_t i = 0; i < elementNums; i++)
	{
		if (min > h_mReconInfoData.imageRecon[i])
		{
			min = h_mReconInfoData.imageRecon[i];
		}
	}

	// ��һ��
	for (size_t i = 0; i < elementNums; i++)
	{
		//image[i] = (image[i] - min) * (end - start) / (max - min) + start;
		h_mReconInfoData.imageRecon[i] = (h_mReconInfoData.imageRecon[i] - min) / (max - min);  // ����[0, 1].
	}
}

// �Ƚ���ֵ��0, Ȼ����й�һ�� ���� ����0�����ֵ
void CBCTFDKRecon::normalizeDataOptimize()
{
	float max = h_mReconInfoData.imageRecon[0], min = h_mReconInfoData.imageRecon[0];
	// �����ֵ
	size_t elementNums = mImagingSystemInfo.pNumX * mImagingSystemInfo.pNumY * mImagingSystemInfo.pNumZ;
	for (size_t i = 0; i < elementNums; i++)
	{
		if (max < h_mReconInfoData.imageRecon[i])
		{
			max = h_mReconInfoData.imageRecon[i];
		}
	}
	// ��ֵ����
	for (size_t i = 0; i < elementNums; i++)
	{
		if (h_mReconInfoData.imageRecon[i] <  0)
		{
			h_mReconInfoData.imageRecon[i] = 0;
		}
	}

	// ��һ��
	for (size_t i = 0; i < elementNums; i++)
	{
		//image[i] = (image[i] - min) * (end - start) / (max - min) + start;
		//h_mReconInfoData.imageRecon[i] = (h_mReconInfoData.imageRecon[i] - 0) / (max - 0);  // ����[0, 1].
		h_mReconInfoData.imageRecon[i] = h_mReconInfoData.imageRecon[i] / max;  // ����[0, 1].
	}
}

void CBCTFDKRecon::findMaxMinValue(float* max, float* min)
{
	

}

void CBCTFDKRecon::printReconInfo()
{
	cout << "========== �ؽ�ϵͳ��Ϣ >>>>>>>>>" << endl;
	cout << "sdd��" << mImagingSystemInfo.sdd << "  " << "sod��" << mImagingSystemInfo.sod << "  " << "̽������Ԫ��С��" << mImagingSystemInfo.dSize << endl;
	cout << "�ؽ�ͼ���С(pix)��" << mImagingSystemInfo.pNumX << "x" << mImagingSystemInfo.pNumY << "x" << mImagingSystemInfo.pNumZ << endl;
	cout << "�ؽ�ͼ���С(mm)��" << mImagingSystemInfo.imgReconLenX << "x" << mImagingSystemInfo.imgReconLenY << "x" << mImagingSystemInfo.imgReconLenZ << endl;
	cout << "�ؽ�ͼ�����ش�С(mm)��" << mImagingSystemInfo.pSizeX << "x" << mImagingSystemInfo.pSizeY << "x" << mImagingSystemInfo.pSizeZ << endl;
	cout << "һ�ܲɼ�ͶӰ������" << mImagingSystemInfo.views << endl;
	cout << "ͶӰ�ߴ磺" << mImagingSystemInfo.dNumU << "x" << mImagingSystemInfo.dNumV << "x" << mImagingSystemInfo.views << endl << endl;
}






