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
#include <map>

using std::vector;
using std::map;

enum PoseEstimationMethod{ ByHomography, ByRansacPnP, Fail };

class FrameMetaData
{
public:
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	clock_t timeStamp;
	char state;
	PoseEstimationMethod method;
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
	std::vector<int> ptIdx;		//	index of 3D points which are in the keyframe
	//map<int, int> coresMap;
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

void CreateKeyFrame(MyMatrix &K, FrameMetaData &currFrame, cv::Mat &currFrameMat, std::vector<KeyFrame> &keyFrames);

void FindNeighboringKeyFrames(std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(KeyFrame &back, FrameMetaData &currFrame, vector <Measurement> &measurementData);

#endif