#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <vector>
#include "KeyFrame.h"
#include "PoseEstimation.h"
using std::vector;

PoseEstimationMethod EstimationMethod();
void CreateFeatureMaps(FeatureMap &featureMap, unsigned int minHessian);
void LoadFeatureMaps(int argc, char *argv[]);
void StopMultiThread();

// Run visual odometry process
bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameImg, cv::Mat &currFrameImg);

#endif