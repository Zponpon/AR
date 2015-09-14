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

//extern std::vector<MyMatrix> projMatrixSet;

class Frame
{
public:
	std::vector<cv::KeyPoint> keypoints;	//Record all keypoints in the image
	cv::Mat descriptors;	//Record all descriptors in the image
	clock_t timeStamp;
	bool state; //不一定需要
	MyMatrix R;
	Vector3d t;
};

class KeyFrame
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	MyMatrix projMatrix;//不一定需要
	MyMatrix R;
	Vector3d t;

	std::vector<cv::KeyPoint> keypoints_3D;
	cv::Mat descriptors_3D;
	std::vector<cv::Point3f> r3dPts;
	unsigned long index;
};

void CreateKeyFrame(unsigned long index, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames, double *cameraPara);

void FindGoodKeyFrames(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, std::vector<int> &goodKeyFrameIdx);

bool KeyFrameSelection(unsigned long index, MyMatrix &R, Vector3d t, std::vector<KeyFrame> &keyFrames);
#endif