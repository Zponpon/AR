#ifndef FRAME_H
#define FRAME_H
#define PI 3.141592653589793
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "Matrix\MyMatrix.h"
#include "BasicType.h"
using std::vector;

class Frame
{
public:
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	clock_t timeStamp;
	MyMatrix R;
	Vector3d t;
};

class EstimateCamInfos
{
public:
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	clock_t timeStamp;
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
	std::vector<cv::KeyPoint> keypoints_3D;
	cv::Mat descriptors_3D;
	std::vector<int> coresIdx; // index of correspondences in keyframe(3D-2D)
	std::vector<cv::Point3d> r3dPts;
	//Matrix & Vector
	MyMatrix projMatrix;//不一定需要
	MyMatrix R;
	Vector3d t;
};

//extern std::vector<KeyFrame> keyFrames;
extern vector<Frame> frameInfos;

void CreateKeyFrame(double *cameraPara, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames);

void FindMatchedKeyFrames(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(std::vector<KeyFrame> &keyFrames, Frame &currFrame);
#endif