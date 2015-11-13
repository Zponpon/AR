#ifndef KEYFRAME_H
#define KEYFRAME_H
#define PI 3.141592653589793
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "Matrix\MyMatrix.h"
#include "BasicType.h"
using std::vector;

class FrameMetaData
{
public:
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	clock_t timeStamp;
	char state;
	MyMatrix R;
	Vector3d t;
};

class KeyFrame
{
public:
	// 2D data	
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	//	3D map	
	std::vector<int> coresIdx; //	correspondence of index in keyframe(3D-2D)
	std::vector<cv::Point3d> r3dPts;

	//	Matrix
	MyMatrix projMatrix;
	MyMatrix R;
	Vector3d t;
};

struct Measurement
{
	//	Record
	double angle;
	double distance;
};

void CreateKeyFrame(MyMatrix &K, FrameMetaData &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames);

void FindNeighboringKeyFrames(std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(MyMatrix &K, KeyFrame &back, FrameMetaData &currFrame, vector <Measurement> &measurementData);

#endif