#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include "KeyFrame.h"
#include "SFMUtil.h"
#include "debugfunc.h"
using std::cout;
using std::endl;

unsigned int keyFrameIdx = 1;


void CreateProjMatrix(MyMatrix &K, const MyMatrix &R, const Vector3d &t, MyMatrix &projMatrix)
{
	/*MyMatrix K(3, 3);
	for (int i = 0; i < 9; ++i)
		K.m_lpdEntries[i] = cameraPara[i];*/
	projMatrix.CreateMatrix(3, 4);
	projMatrix.m_lpdEntries[0] =  R.m_lpdEntries[0];
	projMatrix.m_lpdEntries[1] =  R.m_lpdEntries[1];
	projMatrix.m_lpdEntries[2] =  R.m_lpdEntries[2];
	projMatrix.m_lpdEntries[3] =  t.x;
	projMatrix.m_lpdEntries[4] =  R.m_lpdEntries[3];
	projMatrix.m_lpdEntries[5] =  R.m_lpdEntries[4];
	projMatrix.m_lpdEntries[6] =  R.m_lpdEntries[5];
	projMatrix.m_lpdEntries[7] =  t.y;
	projMatrix.m_lpdEntries[8] =  R.m_lpdEntries[6];
	projMatrix.m_lpdEntries[9] =  R.m_lpdEntries[7];
	projMatrix.m_lpdEntries[10] = R.m_lpdEntries[8];
	projMatrix.m_lpdEntries[11] = t.z;

	projMatrix = K * projMatrix;
}

void CreateKeyFrame(MyMatrix &K, FrameMetaData &currData, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames)
{
	KeyFrame keyFrame;
	currFrameImg.copyTo(keyFrame.image);
	currData.keypoints.swap(keyFrame.keypoints);
	currData.descriptors.copyTo(keyFrame.descriptors);
	keyFrame.R.CreateMatrix(3, 3);
	keyFrame.R = currData.R;
	keyFrame.t = currData.t;
	CreateProjMatrix(K, keyFrame.R, keyFrame.t, keyFrame.projMatrix);
	keyFrames.push_back(keyFrame);
	std::stringstream fileNameStream;
	std::string fileName;
	fileNameStream << "KeyFrame" << keyFrameIdx++ << ".jpg";
	fileName = fileNameStream.str();
	SavingKeyFrame(fileName, keyFrame.image);
}

double calcDistance(Vector3d &t1, Vector3d &t2)
{
	double t1Length = sqrt(pow(t1.x, 2.0) + pow(t1.y, 2.0) + pow(t1.z, 2.0));
	double t2Length = sqrt(pow(t2.x, 2.0) + pow(t2.y, 2.0) + pow(t2.z, 2.0));
	double Cosine = (t1.x*t2.x + t1.y*t2.y + t1.z*t2.z) / (t1Length*t2Length);
	double theta = acos(Cosine) * 180 / PI;
	if (theta > 90)
		return 0.00;

	double distance = sqrt(pow(t1Length, 2.0) + pow(t2Length, 2.0) - 2 * t1Length*t2Length*Cosine);

	return distance;
}

double calcAngle(MyMatrix &K, MyMatrix &R1, MyMatrix &R2)
{
	//相機方向
	Vector3d R1Col2, R2Col2;
	MyMatrix KR1(3, 3), KR2(3, 3);
	KR1 = K*R1;
	KR2 = K*R2;
	double det1 = KR1.Determine();
	double det2 = KR2.Determine();
	R1Col2.x = det1*R1.m_lpdEntries[2];
	R1Col2.y = det1*R1.m_lpdEntries[5];
	R1Col2.z = det1*R1.m_lpdEntries[8];
	R2Col2.x = det2*R2.m_lpdEntries[2];
	R2Col2.y = det2*R2.m_lpdEntries[5];
	R2Col2.z = det2*R2.m_lpdEntries[8];

	double R1Length = sqrt(R1Col2.x*R1Col2.x + R1Col2.y*R1Col2.y + R1Col2.z*R1Col2.z);
	double R2Length = sqrt(R2Col2.x*R2Col2.x + R2Col2.y*R2Col2.y + R2Col2.z*R2Col2.z);
	double Cosine = (R1Col2.x*R2Col2.x + R1Col2.y*R2Col2.y + R1Col2.z*R2Col2.z)/(R1Length*R2Length);
	double angle = acos(Cosine) * 180 / PI;
	cout <<"Angle :"<< Cosine << " "<< acos(Cosine) << " " << angle << endl;

	return angle;
}

void WorldToCamera(MyMatrix &R, Vector3d &t, cv::Point3d &r3dPt, Vector3d &r3dVec)
{
	//To world coordinate to camera coordinate

	MyMatrix pt(3, 1); 
	MyMatrix invR(3, 3);

	//Rotation
	R.Inverse(&invR);
	pt = invR * pt;

	//Translation
	pt.m_lpdEntries[0] = r3dPt.x - t.x;
	pt.m_lpdEntries[1] = r3dPt.y - t.y;
	pt.m_lpdEntries[2] = r3dPt.z - t.z;

	//Vector
	r3dVec.x = pt.m_lpdEntries[0];
	r3dVec.y = pt.m_lpdEntries[1];
	r3dVec.z = pt.m_lpdEntries[2];
}

bool isNegihboringKeyFrame(Vector3d &t1, Vector3d &t2, Vector3d &r3dVec1, Vector3d &r3dVec2)
{
	//用兩張frame的原點算出原點的3D點座標
	//計算t跟原點跟3D點座標的向量之夾角
	
	//cout << "3D point : " << r3dPtVec.x << ", " << r3dPtVec.y << ", " << r3dPtVec.z << endl;
	double r3dVec1Length = sqrt(r3dVec1.x*r3dVec1.x + r3dVec1.y*r3dVec1.y + r3dVec1.z*r3dVec1.z);
	double t1Length = sqrt(t1.x*t1.x + t1.y*t1.y + t1.z*t1.z);
	double Cosine1 = (t1.x*r3dVec1.x + t1.y*r3dVec1.y + t1.z*r3dVec1.z) / (r3dVec1Length*t1Length);

	double r3dVec2Length = sqrt(r3dVec2.x*r3dVec2.x + r3dVec2.y*r3dVec2.y + r3dVec2.z*r3dVec2.z);
	double t2Length = sqrt(t2.x*t2.x + t2.y*t2.y + t2.z*t2.z);
	double Cosine2 = (t2.x*r3dVec2.x + t2.y*r3dVec2.y + t2.z*r3dVec2.z) / (r3dVec2Length*t2Length);

	double theta1 = acos(Cosine1) * 180 / PI;
	double theta2 = acos(Cosine2) * 180 / PI;
	
	if (theta1 > 90.0 || theta2 > 90.0)
		return false;
	if (abs(theta1 - theta2) >= 30.0)
		return false;
	return true;
}

void FindNeighboringKeyFrames(std::vector<KeyFrame> &keyFrames, FrameMetaData &currData, std::vector<int> &neighboringKeyFrameIdx)
{
	/*	Use the last keyframe to find the neighboring keyframes	*/
	//用最新的一張keyframe去找尋整個set
	//並找出跟最後一張keyframe相鄰的keyframe
	int keyFramesSize = (int)keyFrames.size() - 1;
	cv::Point2f originPt(400.0f, 300.0f);
	int index = 0;
	for (std::vector<KeyFrame>::iterator KF = keyFrames.begin(); index < keyFramesSize; ++index)
	{
		cv::Point3d r3dPt;
		if (Find3DCoordinates(KF->projMatrix, keyFrames.back().projMatrix, originPt, originPt, r3dPt))
		{
			Vector3d r3dVec1, r3dVec2;
			WorldToCamera(KF->R, KF->t, r3dPt, r3dVec1);
			WorldToCamera(keyFrames.back().R, keyFrames.back().t, r3dPt, r3dVec2);

			if (isNegihboringKeyFrame(KF->t, keyFrames.back().t, r3dVec1, r3dVec2))
				neighboringKeyFrameIdx.push_back(index);
		}
	}
	//	Push the last keyframe
	neighboringKeyFrameIdx.push_back(keyFramesSize);
}

bool KeyFrameSelection(MyMatrix &K, KeyFrame &keyFramesBack, FrameMetaData &currData, vector <Measurement> &measurementData)
{
	if (currData.state == 'I')
		return false;
	double distance = calcDistance(keyFramesBack.t, currData.t);
	if(distance < 150.0 || isnan(distance))
		return false;
	double angle = calcAngle(K, keyFramesBack.R, currData.R);
	if (angle < 30.0f || isnan(angle))
		return false;

	Measurement measurement;
	measurement.distance = distance;
	measurement.angle = angle;
	measurementData.push_back(measurement);
	return true;
}