#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include <iostream>
#include <vector>
#include "SiftGPU.h"
#include "KeyFrame.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"


class FeatureMap
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

void CreateFeatureMap(FeatureMap &featureMap, int minHessian);

//For homography
void EstimateCameraTransformation(double *cameraPara, double trans[3][4], FeatureMap &featureMap, FrameMetaData &currFrame, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers);

//For Visual odometry
void EstimateCameraTransformation(double *cameraPara, double trans[3][4], std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame, std::vector<int> &neighboringKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet);

#endif