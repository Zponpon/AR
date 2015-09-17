#ifndef FRAME_H
#define FRAME_H
#define PI 3.141592653589793
#include <vector>
#include <cmath>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"
#include "Matrix\MyMatrix.h"
#include "BasicType.h"

class Frame
{
public:
	std::vector<cv::KeyPoint> keypoints;
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
	std::vector<cv::Point3d> r3dPts;
	//Matrix & Vector
	MyMatrix projMatrix;//不一定需要
	MyMatrix R;
	Vector3d t;
};

void CreateKeyFrame(double *cameraPara, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames);

void FindMatchedKeyFrames(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(std::vector<KeyFrame> &keyFrames, Frame &currFrame);
#endif