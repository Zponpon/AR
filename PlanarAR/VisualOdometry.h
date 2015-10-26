#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <vector>
#include "KeyFrame.h"
#include "PoseEstimation.h"
using std::vector;

void CreateFeatureMaps(FeatureMap &featureMap, unsigned int minHessian);
void LoadFeatureMaps(int argc, char *argv[]);
void StopMultiThread();
char EstimationMethod();
bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameImg, cv::Mat &currFrameImg);

#endif