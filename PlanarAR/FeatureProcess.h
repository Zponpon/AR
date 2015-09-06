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
#include "debugfunc.h"

using std::cout;
using std::endl;
using cv::imshow;
using cv::waitKey;
using cv::imwrite;

void CreateFrame(Frame &frame, unsigned int minHessian);
void SurfDetection(cv::Mat &data, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian);
void FlannMatching(cv::Mat &descriptors_img, cv::Mat &descriptors_scene, std::vector<cv::DMatch> &matches);
void OpticalFlow(cv::Mat preFrame, cv::Mat &curFrame, std::vector<cv::Point2f> &preFrameGoodMatches, std::vector<cv::Point2f> &preImageGoodMatches, std::vector<cv::Point2f> &img_goodmatches, std::vector<cv::Point2f> &scene_goodmatches);
void OpticalFlow(FeatureMap &featureMap, KeyFrame &prevFrame, Frame &currFrame, std::vector<cv::Point2f> &currPts, std::vector<int> &good3DMatches);
void DeleteOverlap(std::vector<cv::Point2f> &img_goodmatches, std::vector<cv::Point2f> &scene_goodmatches);
bool FeatureDetectionAndMatching(FeatureMap &featureMap, Frame &prevFrame, Frame &currFrame, unsigned int minHessian, std::vector<cv::Point2f>  &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &prevFrameGoodMatches);
//刪除重複對應的特徵點
void FindGoodMatches(FeatureMap &featureMap, Frame &frame, std::vector<cv::KeyPoint> &keypoints_scene, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &img_goodmatches, std::vector<cv::Point2f> &scene_goodmatches);
//For triangulate points
void FindGoodMatches(KeyFrame &img1, KeyFrame &img2, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &img1_goodMatches, std::vector<cv::Point2f> &img2_goodMatches, std::vector<cv::KeyPoint> &keypoints_3D, cv::Mat &descriptors_3D);
//For 3D points
void FindGoodMatches(Frame &img1, Frame &img2, std::vector<cv::DMatch> &matches, std::vector<int> &good3DMatches, std::vector<cv::Point2f> &img2_goodMatches);

#endif