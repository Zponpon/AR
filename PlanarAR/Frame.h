#ifndef FRAME_H
#define FRAME_H
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2\stitching\stitcher.hpp"


class Frame
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
};

class KeyFrame
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint> feature;
	cv::Mat descriptors;
	cv::Mat projMatrix;
	unsigned long index;
	//std::vector <cv::Point3f> feature3dPts;
};

bool KeyFrameSelection(unsigned long index, std::vector<KeyFrame> &keyFrames);
#endif