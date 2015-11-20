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
	KeyFrame keyframe;

	currFrameImg.copyTo(keyframe.image);
	currData.keypoints.swap(keyframe.keypoints);
	currData.descriptors.copyTo(keyframe.descriptors);

	keyframe.R.CreateMatrix(3, 3);
	keyframe.R = currData.R;
	keyframe.t = currData.t;
	CreateProjMatrix(K, keyframe.R, keyframe.t, keyframe.projMatrix);

	keyFrames.push_back(keyframe);
	std::stringstream fileNameStream;
	std::string fileName;
	fileNameStream << "KeyFrame" << keyFrameIdx++ << ".jpg";
	fileName = fileNameStream.str();
	SavingKeyFrame(fileName, keyframe.image);
	
}

double calcDistance(Vector3d &t1, Vector3d &t2)
{
	return sqrt((t1.x - t2.x)*(t1.x - t2.x) + (t1.y - t2.y)*(t1.y - t2.y) + (t1.z - t2.z)*(t1.z - t2.z));
}

double calcAngle(MyMatrix &K, MyMatrix &R1, MyMatrix &R2)
{
	//相機方向
	Vector3d R1Col2, R2Col2;

	//R3反方向
	//不需要乘上K
	//因為我們是以
	R1Col2.x = R1.m_lpdEntries[2]*(-1);
	R1Col2.y = R1.m_lpdEntries[5]*(-1);
	R1Col2.z = R1.m_lpdEntries[8]*(-1);
	R2Col2.x = R2.m_lpdEntries[2]*(-1);
	R2Col2.y = R2.m_lpdEntries[5]*(-1);
	R2Col2.z = R2.m_lpdEntries[8]*(-1);

	double R1Length = sqrt(R1Col2.x*R1Col2.x + R1Col2.y*R1Col2.y + R1Col2.z*R1Col2.z);
	double R2Length = sqrt(R2Col2.x*R2Col2.x + R2Col2.y*R2Col2.y + R2Col2.z*R2Col2.z);
	double Cosine = (R1Col2.x*R2Col2.x + R1Col2.y*R2Col2.y + R1Col2.z*R2Col2.z)/(R1Length*R2Length);
	double angle = acos(Cosine) * 180 / PI;

	return angle;
}

bool KeyFrameSelection(MyMatrix &K, KeyFrame &keyFramesBack, FrameMetaData &currData, vector <Measurement> &measurementData)
{
	double distance = calcDistance(keyFramesBack.t, currData.t);
	if (distance <= 250.0f || isnan(distance))
	{
		cout << "Distance : " << distance << endl;
		return false;
	}

	double angle = calcAngle(K, keyFramesBack.R, currData.R);
	if (angle <= 30.0f || isnan(angle))
	{
		cout << "Angle : " << angle << endl;
		return false;
	}

	Measurement measurement;
	measurement.distance = distance;
	measurement.angle = angle;
	measurementData.push_back(measurement);

	return true;
}


/*	Finding neighboring keyframe	*/

void WorldToCamera(MyMatrix &R, Vector3d &t, cv::Point3d &r3dPt, Vector3d &r3dVec)
{
	//From world coordinate to camera coordinate

	MyMatrix pt(3, 1);
	//Rotation R
	pt.m_lpdEntries[0] = r3dPt.x;
	pt.m_lpdEntries[1] = r3dPt.y;
	pt.m_lpdEntries[2] = r3dPt.z;
	pt = R * pt;

	//Translation t'
	pt.m_lpdEntries[0] = r3dPt.x+t.x;
	pt.m_lpdEntries[1] = r3dPt.y+t.y;
	pt.m_lpdEntries[2] = r3dPt.z+t.z;

	//Vector
	r3dVec.x = pt.m_lpdEntries[0];
	r3dVec.y = pt.m_lpdEntries[1];
	r3dVec.z = pt.m_lpdEntries[2];
}

bool isNegihboringKeyFrame(Vector3d &t1, Vector3d &t2, Vector3d &r3dVec1, Vector3d &r3dVec2)
{
	//用兩張frame的原點算出原點的3D點座標
	//計算t跟原點跟3D點座標的向量之夾角

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
	/*	Use the last keyframe for finding the neighboring keyframes	*/
	//目前全部比對
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
		//neighboringKeyFrameIdx.push_back(index);
	}
	//	Push the last keyframe
	neighboringKeyFrameIdx.push_back(keyFramesSize);
}
