#ifndef DEBUGFUNC_H
#define DEBUGFUNC_H
//#define SAVEIMAGE
//#define PRINTTHE3DPOINTSDATA

#include "SiftGPU.h"
#include "string"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"
#include <fstream>
#include <vector>


using std::iostream;
using std::vector;

void DrawPoint(unsigned char *Data, int Width, int Height, int x, int y, unsigned char r, unsigned char g, unsigned char b);
void DebugMarkPoint(unsigned char* Data, int Width, int Height, vector<SiftGPU::SiftKeypoint> keys);
void DebugMarkMatchedPoint(unsigned char* imgData, int Width, int Height, int nMatched,  int (*match_buf)[2], int ImageIndex, vector<cv::KeyPoint> keys);
void DebugSaveCorrespondences(char *filename, vector<SiftGPU::SiftKeypoint> &keys1, vector<SiftGPU::SiftKeypoint> &keys2, int match_buf[][2], int num_match);

void DebugOpenCVMarkPoint(cv::Mat &data, std::vector<cv::KeyPoint> &keypoints, char *name);
void DebugOpenCVMatchPoint(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints_img, cv::Mat &scene, std::vector<cv::KeyPoint> &keypoints_scene, std::vector<cv::DMatch> &good_matches, char *name);

void DebugSaveImage(char *filename, unsigned char *pbImage, int iWidth, int iHeight, int nChannel);
void DebugLoadImage(char *filename, unsigned char *pbImage, int iWidth, int iHeight, int nChannel);

void DrawOpticalFlow(std::vector<cv::Point2f> preImg_goodmatches, std::vector<cv::Point2f> preScene_goodmatches, std::vector<cv::Point2f> OpticalFlow_keypoints); //Point2f vector good keypoints in previous frame

void WriteVideo();

void WritePointsFile(std::vector<cv::Point2f>, std::vector<cv::Point2f>);
void WriteRecordFile(unsigned long FrameCount, int GoodMatchesSize, int OpticalFlowKeyPointsSize);
#endif