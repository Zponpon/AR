#include "Frame.h"
#include "SFMUtil.h"

void CreateProjMatrix(double *cameraPara, const MyMatrix &R, const Vector3d &t, MyMatrix &projMatrix)
{
	MyMatrix K(3, 3);
	for (int i = 0; i < 9; ++i)
		K.m_lpdEntries[i] = cameraPara[i];
	projMatrix.m_lpdEntries[0] = R.m_lpdEntries[0];
	projMatrix.m_lpdEntries[1] = R.m_lpdEntries[1];
	projMatrix.m_lpdEntries[2] = R.m_lpdEntries[2];
	projMatrix.m_lpdEntries[3] = t.x;
	projMatrix.m_lpdEntries[4] = R.m_lpdEntries[3];
	projMatrix.m_lpdEntries[5] = R.m_lpdEntries[4];
	projMatrix.m_lpdEntries[6] = R.m_lpdEntries[5];
	projMatrix.m_lpdEntries[7] = t.y;
	projMatrix.m_lpdEntries[8] = R.m_lpdEntries[6];
	projMatrix.m_lpdEntries[9] = R.m_lpdEntries[7];
	projMatrix.m_lpdEntries[10] = R.m_lpdEntries[8];
	projMatrix.m_lpdEntries[11] = t.z;

	projMatrix = K * projMatrix;
}

void CreateKeyFrame(double *cameraPara, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames)
{
	KeyFrame keyFrame;
	currFrameImg.copyTo(keyFrame.image);
	currFrame.keypoints.swap(keyFrame.keypoints);
	currFrame.descriptors.copyTo(keyFrame.descriptors);
	keyFrame.R.CreateMatrix(3, 3);
	keyFrame.R = currFrame.R;
	keyFrame.t = currFrame.t;
	CreateProjMatrix(cameraPara, keyFrame.R, keyFrame.t, keyFrame.projMatrix);
	keyFrames.push_back(keyFrame);
}

double calcDistance(Vector3d &t1, Vector3d &t2)
{
	double t1Length = sqrt(pow(t1.x, 2.0) + pow(t1.y, 2.0) + pow(t1.z, 2.0));
	double t2Length = sqrt(pow(t2.x, 2.0) + pow(t2.y, 2.0) + pow(t2.z, 2.0));
	double Cosine = (t1.x*t2.x + t1.y*t2.y + t1.z*t2.z) / (t1Length*t2Length);
	//double theata = acos(Cosine);
	double distance = sqrt(pow(t1Length, 2.0) + pow(t2Length, 2.0) - 2 * t1Length*t2Length*Cosine);
	return distance;
}

double calcAngle(MyMatrix &R1, MyMatrix &R2)
{
	//相機方向
	Vector3d R1Col2, R2Col2;
	R1Col2.x = R1.m_lpdEntries[2] * (-1);
	R1Col2.y = R1.m_lpdEntries[5] * (-1);
	R1Col2.z = R1.m_lpdEntries[8] * (-1);
	R2Col2.x = R2.m_lpdEntries[2] * (-1);
	R2Col2.y = R2.m_lpdEntries[5] * (-1);
	R2Col2.z = R2.m_lpdEntries[8] * (-1);
	
	double R1Col2Length = sqrt(pow(R1Col2.x, 2.0) + pow(R1Col2.y, 2.0) + pow(R1Col2.z, 2.0));
	double R2Col2Length = sqrt(pow(R2Col2.x, 2.0) + pow(R2Col2.y, 2.0) + pow(R2Col2.z, 2.0));
	double Cosine = (R1Col2.x*R2Col2.x + R1Col2.y*R2Col2.y + R1Col2.z*R1Col2.z) / (R1Col2Length*R2Col2Length);
	double theta = acos(Cosine);
	
	return theta;
}

bool isMatchedKeyFrame(Vector3d &t1, Vector3d &t2, Vector3d &r3dPtVec)
{
	//用兩張frame的原點算出原點的3D點座標
	//計算t跟原點跟3D點座標的向量之夾角
	//
	double r3dPtVecLength = sqrt(pow(r3dPtVec.x, 2.0) + pow(r3dPtVec.y, 2.0) + pow(r3dPtVec.z, 2.0));
	double t1Length = sqrt(pow(t1.x, 2.0) + pow(t1.y, 2.0) + pow(t1.z, 2.0));
	double t2Length = sqrt(pow(t2.x, 2.0) + pow(t2.y, 2.0) + pow(t2.z, 2.0));
	double Cosine1 = (t1.x*r3dPtVec.x + t1.y*r3dPtVec.y + t1.z*r3dPtVec.z) / (r3dPtVecLength*t1Length);
	double Cosine2 = (t2.x*r3dPtVec.x + t2.y*r3dPtVec.y + t2.z*r3dPtVec.z) / (r3dPtVecLength*t2Length);
	double theta1 = acos(Cosine1);
	double theta2 = acos(Cosine2);
	if (theta1 > 90.0)
		return false;
	//Test？
	if (theta1 < theta2 + 30.0)
		return true;
	return false;
}

void FindMatchedKeyFrames(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, std::vector<int> &goodKeyFrameIdx)
{
	//SolvePnPRansac
	MyMatrix projMatrix(3, 4);
	CreateProjMatrix(cameraPara, currFrame.R, currFrame.t, projMatrix);
	cv::Point2f originPt(400.0f, 300.0f);
	for (int i = 0; i < keyFrames.size(); ++i)
	{
		cv::Point3d r3dPt;
		if (Find3DCoordinates(keyFrames[i].projMatrix, projMatrix, originPt, originPt, r3dPt))
		{
			Vector3d r3dPtVec;
			r3dPtVec.x = (double)r3dPt.x; r3dPtVec.y = (double)r3dPt.y; r3dPtVec.z = (double)r3dPt.z;
			if (isMatchedKeyFrame(keyFrames[i].t, currFrame.t, r3dPtVec))
				goodKeyFrameIdx.push_back(i);
		}
	}
}

bool KeyFrameSelection(std::vector<KeyFrame> &keyFrames, Frame &currFrame)
{
	//if(calcDistance < xxx)
	return false;
	//if(calcAngle < xx)
	return false;

	return true;
}