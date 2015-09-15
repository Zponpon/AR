#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include <iostream>
#include <vector>
#include "SiftGPU.h"
#include "Frame.h"
#include "opencv2\opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"


class FeatureMap
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
//	std::vector<cv::KeyPoint> reProjPts;//不要的
	std::vector<cv::Point3f> feature3D; 		//不要的															
};
void CreateFeatureMap(FeatureMap &featureMap, int minHessian);
//void Triangulation(std::vector<KeyFrame> &keyFrames, FeatureMap &featureMap, double *cameraPara);

//For homography
bool EstimateCameraTransformation(unsigned long FrameCount, double *cameraPara, double trans[3][4], FeatureMap &featureMap, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers);
//bool EstimateCameraTransformation(unsigned long FrameCount, std::vector<KeyFrame> &keyFrames, unsigned char *inputprevFrame, unsigned char **inputFrame, int frameWidth, int frameHeight, FeatureMap &featureMap, double *cameraPara, double trans[3][4]);
//For Visual odometry
bool EstimateCameraTransformation(double *cameraPara, double trans[3][4], std::vector<KeyFrame> &keyFrames, Frame &currFrame, cv::Mat &currFrameImg, std::vector<int> &goodKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet);

#endif