#include "Frame.h"

//std::vector<MyMatrix> projMatrixSet;
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

bool KeyFrameSelection(unsigned long index, MyMatrix &R, Vector3d t, std::vector<KeyFrame> &keyFrames)
{
	if (index - keyFrames[0].index < 20)
		return false;
	//std::size_t startingIdx = 0;
	//Test three keyframes¡H
	for (std::size_t i = keyFrames.size() - 3; i < keyFrames.size();)
	{
		//What is the threshold¡H
		if (calcDistance(keyFrames[i].t, t) > 50 || calcAngle(keyFrames[i].R, R) < 30)
			return true;
		else
		{
			++i;
			if (i == keyFrames.size())
				return false;
		}
	}

	//if (calcDistance(keyFrames[0]))
	return true;
}