#define FEATUREPROCESS_H
#ifdef  FEATUREPROCESS_H
//#define SHOWTHEIMAGE
#include <iostream>
#include <windows.h>
#include <GL/gl.h>
#include "glut.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"
#include "PoseEstimation.h"
#include "debugfunc.h"

using std::cout;
using std::endl;
using cv::imshow;
using cv::waitKey;
using cv::imwrite;

void SurfDetection(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian);
void FlannMatching(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches);
bool FeatureDetection(unsigned int minHessian, Frame &currFrame, cv::Mat &currFrameImg);

//For homography
bool FeatureMatching(FeatureMap &featureMap, Frame &currFrame, cv::Mat &currFrameImg, cv::Mat &prevFrameImg, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &prevFrameGoodMatches);

//For solvePnPRansac
bool FeatureMatching(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, cv::Mat &currFrameImg, std::vector<int> &goodKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet);

//For triangulation (Two cases)
void FindGoodMatches(KeyFrame &KF1, KeyFrame &KF2, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &KF1GoodMatches, std::vector<cv::Point2f> &KF2GoodMatches, std::vector<cv::KeyPoint> *keypointsMatches, cv::Mat *descriptorsMatches);

#endif