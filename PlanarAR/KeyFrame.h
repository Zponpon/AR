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
	char state[1];
	MyMatrix R;
	Vector3d t;
};

class KeyFrame
{
public:
	//2D
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	//3D map
	/*
	std::vector<cv::KeyPoint> keypoints_3D;
	cv::Mat descriptors_3D;
	*/
	std::vector<int> coresIdx; // index of correspondences in keyframe(3D-2D)
	std::vector<cv::Point3d> r3dPts;
	//Matrix & Vector
	MyMatrix projMatrix;//不一定需要
	MyMatrix R;
	Vector3d t;
};

//extern std::vector<KeyFrame> keyFrames;
//extern vector<FrameM> frameInfos;

void CreateKeyFrame(double *cameraPara, FrameMetaData &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames);

void FindNeighboringKeyFrames(std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame);
#endif