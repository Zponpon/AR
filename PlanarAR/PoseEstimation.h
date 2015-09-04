#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include <iostream>
#include <vector>
#include "SiftGPU.h"
#include "Frame.h"
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
	std::vector<cv::KeyPoint> keypoints;	//Feature in the featureMap
	cv::Mat descriptors;

	std::vector<cv::KeyPoint> reProjPts;
	std::vector<cv::Point3f> feature3D; 
	/*****************************************************************************/
	/*ReproPts are the 3D point project to the image                             */
	/*feature3D record the features in the world coordinate are not on the marker*/																		
};	/*****************************************************************************/						
						
/*class Frame
{
public:
	void release()
	{
		image.release();
		std::vector<cv::KeyPoint>().swap(keypoints);
		descriptors.release();
		std::vector<cv::KeyPoint>().swap(keypoints_3D);
		descriptors_3D.release();
		projMatrix.release();
	}
	Frame& operator= (Frame &frame)
	{
		frame.image.copyTo(image);
		keypoints = frame.keypoints;
		frame.descriptors.copyTo(descriptors);
		keypoints_3D = frame.keypoints_3D;
		frame.descriptors_3D.copyTo(descriptors_3D);
		frame.projMatrix.copyTo(projMatrix);
		return *this;
	}
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;	//Record all keypoints in the image
	cv::Mat descriptors;	//Record all descriptors in the image
	std::vector<cv::KeyPoint> keypoints_3D; //3D feature's keypoint(2D) in the image
	cv::Mat descriptors_3D; //Record the 3D points descriptors in the 2D image
	cv::Mat projMatrix;//The projection matrix is from world coordinate to image coordinate
};*/

void CreateFeatureMap(FeatureMap &featureMap, int minHessian);
void Triangulation(Frame &frame1, Frame &frame2, FeatureMap &featureMap, double *cameraPara);
//For homography
bool EstimateCameraTransformation(std::vector<Frame > &keyFrames, unsigned char *inputprevFrame, unsigned char **inputFrame, int frameWidth, int frameHeight, FeatureMap &featureMap, double *cameraPara, double trans[3][4]);
//For Visual odometry
bool EstimateCameraTransformation(FeatureMap &featureMap, Frame &prevImgTemp, Frame &previmg, unsigned char *inputFrame, int frameWidth, int frameHeight, double *cameraPara, double trans[3][4]);

#endif