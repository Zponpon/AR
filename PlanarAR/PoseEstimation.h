#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include <iostream>
#include <vector>
#include "SiftGPU.h"
#include "KeyFrame.h"
#include "SFMUtil.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/features2d/features2d.hpp"

class FeatureMap
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

/*	Method : Homography(OpenCV)	*/
void EstimateCameraTransformation(double *cameraPara, double trans[3][4], FeatureMap &featureMap, FrameMetaData &currFrame, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers);

/*	Estimation method : PnP Ransac(OpenCV)	*/
void EstimateCameraTransformation(MyMatrix &cameraMatrix, double trans[3][4], std::vector<cv::Point3d> &r3dPts, std::vector<KeyFrame> &keyFrames, FrameMetaData &currFrame, std::vector<int> &neighboringKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet);

#endif