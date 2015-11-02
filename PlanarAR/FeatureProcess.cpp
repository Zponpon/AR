#include "FeatureProcess.h"

/*	Detection method : Surf(OpenCV)	*/
void SurfDetection(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian)
{
	int win = glutGetWindow();
	cv::SurfFeatureDetector detector(minHessian);
	detector.detect(image, keypoints);
	if (keypoints.size() == 0){
		glutSetWindow(win);
		return;
	}
	cv::SurfDescriptorExtractor extractor;
	extractor.compute(image, keypoints, descriptors);

	glutSetWindow(win);
}

void FlannMatching(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
	int win = glutGetWindow();
	cv::FlannBasedMatcher matcher;
	if (descriptors1.type() != CV_32F)
		descriptors1.convertTo(descriptors1, CV_32F);
	if (descriptors2.type() != CV_32F)
		descriptors2.convertTo(descriptors2, CV_32F);
	matcher.match(descriptors1, descriptors2, matches);
	glutSetWindow(win);
}

bool FeatureDetection(FeatureMap &featureMap, unsigned int minHessian)
{
	//Detect keypoints of featureMaps only
	SurfDetection(featureMap.image, featureMap.keypoints, featureMap.descriptors, minHessian);
	if (featureMap.keypoints.size() == 0)
		return false;
	return true;
}

bool FeatureDetection(unsigned int minHessian, FrameMetaData &currData, cv::Mat &currFrameImg)
{
	SurfDetection(currFrameImg, currData.keypoints, currData.descriptors, minHessian);
	if (currData.keypoints.size() == 0)
		return false;
	return true;
}

void RemoveDuplicatePts(std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches)
{
	//	把標記為True的點放入GoodMatchedPoints
	//	刪除OpticalFlow和Surf重複的特徵點

	std::vector<cv::Point2f> GoodMatchedPoints[2];	//[0]->featureMap(marker)中沒有重複的點, [1]->frame中沒有重複的點
	std::vector<bool> NotOverlayPointsFlag(featureMapGoodMatches.size(), true);	//標記keypoints x & y一樣的點為False

	for (std::size_t i = 0; i < featureMapGoodMatches.size(); i++)
	{
		if (!NotOverlayPointsFlag[i])
			continue;
		for (std::size_t j = i + 1; j < featureMapGoodMatches.size(); j++)
		{
			float errX = std::abs(frameGoodMatches[i].x - frameGoodMatches[j].x);
			float errY = std::abs(frameGoodMatches[i].y - frameGoodMatches[j].y);
			if (!NotOverlayPointsFlag[j])
				continue;
			if (errX <= 1.0 && errY <= 1.0)	//誤差值該設定為多少??
				NotOverlayPointsFlag[j] = false;
		}
	}

	//位置接近的特徵點暫時先刪除OpticalFlow算出來的
	for (size_t i = 0; i < NotOverlayPointsFlag.size(); i++)
	{
		if (NotOverlayPointsFlag[i])
		{
			GoodMatchedPoints[0].push_back(featureMapGoodMatches[i]);
			GoodMatchedPoints[1].push_back(frameGoodMatches[i]);
		}
	}
	featureMapGoodMatches.swap(GoodMatchedPoints[0]);
	frameGoodMatches.swap(GoodMatchedPoints[1]);

	//free vector memory
	std::vector<cv::Point2f>().swap(GoodMatchedPoints[0]);
	std::vector<cv::Point2f>().swap(GoodMatchedPoints[1]);
	std::vector<bool>().swap(NotOverlayPointsFlag);
}

void OpticalFlow(cv::Mat &prevFrame, cv::Mat &currFrame, std::vector<cv::Point2f> &prevFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches)
{
	//Add more feature points
	int GoodSize = 0;
	std::vector<cv::Point2f> GoodMatches[2]; //[0]->preFeatureMap, [1]->prevFrame, store the feature points
	std::vector<uchar> status;	//record OpticalPyrLK keypoints which are right tracking
	std::vector<float> error;	//record OpticalPyrLK error
	std::vector<cv::Point2f> OpticalFlow_keypoints;
	
	// OpenCV function to find OpticalFlow points
	cv::calcOpticalFlowPyrLK(prevFrame, currFrame, prevFrameGoodMatches, OpticalFlow_keypoints, status, error);

	for (std::size_t i = 0; i < OpticalFlow_keypoints.size(); ++i)
	{
		//根據這個code來設定threshold https://github.com/Itseez/opencv/blob/master/samples/cpp/lkdemo.cpp/
		if (!(status[i] == 0 || (cv::norm(OpticalFlow_keypoints[i] - prevFeatureMapGoodMatches[i]) <= 5)))	//目前為經驗法則
		{
			GoodMatches[0].push_back(prevFeatureMapGoodMatches[i]);//避免拍到不是target image時，依舊保留上一個frame的特徵點
			GoodMatches[1].push_back(prevFrameGoodMatches[i]);
			featureMapGoodMatches.push_back(prevFeatureMapGoodMatches[i]);
			frameGoodMatches.push_back(OpticalFlow_keypoints[i]);
			GoodSize++;
		}
	}
	prevFeatureMapGoodMatches.swap(GoodMatches[0]);
	prevFrameGoodMatches.swap(GoodMatches[1]);
	cout << "OpticalFlow Size : " << GoodSize << endl;
	/*

	std::vector<cv::Point2f>().swap(GoodMatches[0]);
	std::vector<cv::Point2f>().swap(GoodMatches[1]);
	std::vector<cv::Point2f>().swap(OpticalFlow_keypoints);
	std::vector<uchar>().swap(status);
	std::vector<float>().swap(error);

	*/
}

/*								*/
/*	Remove the false matching	*/
/*								*/
void FindGoodMatches(FeatureMap &featureMap, cv::Mat &currFrame, std::vector<cv::KeyPoint> &keypoints_frame, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches)
{
	//This method is to delete the feature points which are multimatch
	//傳入currFrame做DEBUG用

	std::vector<cv::DMatch> goodMatches;
	//	Quick calcul ation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "(max, min) : (" << max_dist << ", " << min_dist << ")\n\n";
	//DebugOpenCVMarkPoint(frame, keypoints_frame, "inputFrame.jpg");
	//	Find the good matches
	min_dist *= 2.0;
	if (min_dist > 0.2)
		min_dist = 0.2;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist)
			goodMatches.push_back(matches[i]);
	}
	//標記一對多的點為false-> goodMatches
	std::vector<bool> GoodMatchesFlag(goodMatches.size(), true);	//判斷Surf算出來的點

	for (std::size_t j = 0; j < goodMatches.size(); j++)
	{
		std::size_t index = j;	//紀錄distance較小的index
		double distance = goodMatches[index].distance;
		if (!GoodMatchesFlag[index])
			continue;
		for (std::size_t k = j + 1; k < goodMatches.size(); k++)
		{
			if (!GoodMatchesFlag[k])
				continue;
			if (goodMatches[index].trainIdx == goodMatches[k].trainIdx)
			{
				if (distance <= goodMatches[k].distance)
					GoodMatchesFlag[k] = false;
				else
				{
					GoodMatchesFlag[index] = false;
					index = k;
					distance = goodMatches[index].distance;
				}
			}
		}
	}
	//DebugOpenCVMatchPoint(featureMap.image, featureMap.keypoints, frame.image, frame.keypoints, goodMatches, "IMG1.JPG");
	//把標記為true的點放入featureMap_goodMatches & frame_goodMatches
	for (std::size_t i = 0; i < goodMatches.size(); i++)
	{
		if (GoodMatchesFlag[i])
		{
			featureMapGoodMatches.push_back(featureMap.keypoints[goodMatches[i].queryIdx].pt);
			frameGoodMatches.push_back(keypoints_frame[goodMatches[i].trainIdx].pt);
		}
	}
}

bool FeatureMatching(FeatureMap &featureMap, FrameMetaData &currData, cv::Mat &currFrameImg, cv::Mat &prevFrameImg, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers)
{
	//Matching with the featureMap

	std::vector<cv::DMatch> matches;
	FlannMatching(featureMap.descriptors, currData.descriptors, matches);
	FindGoodMatches(featureMap, currFrameImg, currData.keypoints, matches, featureMapGoodMatches, currFrameGoodMatches);

	//GoodMatches size is not enough
	bool usingOpticalFlow = false;
	if ((int)currFrameGoodMatches.size() < 25)
	{
		if (prevFrameInliers.size() != 0 && prevFeatureMapInliers.size() != 0 && prevFrameImg.data != NULL)
		{
			//Use optical flow to detect feature
			OpticalFlow(prevFrameImg, currFrameImg, prevFrameInliers, prevFeatureMapInliers, featureMapGoodMatches, currFrameGoodMatches);
			usingOpticalFlow = true;
		}
		else
		{
			prevFeatureMapInliers.swap(featureMapGoodMatches);
			prevFrameInliers.swap(currFrameGoodMatches);

			return false;
		}
	}

	if (usingOpticalFlow)
	{
		RemoveDuplicatePts(featureMapGoodMatches, currFrameGoodMatches);
		if ((int)currFrameGoodMatches.size() < 30)
			return false;
	}

	return true;
}

void FindGoodMatches(std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &goodMatches)
{
	double max_dist = 0; double min_dist = 100;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//	Find the good matches
	min_dist *= 2.0;
	if (min_dist > 0.2) min_dist = 0.2;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist)
			goodMatches.push_back(matches[i]);
	}

	//標記一對多的點為false-> goodMatches
	std::vector<bool> GoodMatchesFlag(goodMatches.size(), true);
	for (std::size_t j = 0; j < goodMatches.size(); j++)
	{
		std::size_t index = j;	//紀錄distance較小的index
		double distance = goodMatches[index].distance;
		if (!GoodMatchesFlag[index])
			continue;
		for (std::size_t k = j + 1; k < goodMatches.size(); k++)
		{
			if (!GoodMatchesFlag[k])
				continue;
			if (goodMatches[index].trainIdx == goodMatches[k].trainIdx)
			{
				if (distance <= goodMatches[k].distance)
					GoodMatchesFlag[k] = false;
				else
				{
					GoodMatchesFlag[index] = false;
					index = k;
					distance = goodMatches[index].distance;
				}
			}
		}
	}
	std::vector<cv::DMatch> tmp;
	for (std::vector<cv::DMatch>::size_type i = 0; i < goodMatches.size(); i++)
	{
		if (GoodMatchesFlag[i])
			tmp.push_back(goodMatches[i]);
	}
	goodMatches.swap(tmp);
}

/*	PnP	*/
bool FeatureMatching(double *cameraPara, std::vector<KeyFrame> &keyFrames, FrameMetaData &currData, cv::Mat &currFrameImg, std::vector<int> &neighboringKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet)
{
	/* When we move the camera to the place where are without the featureMap                  */
	/* We use the last keyframe to find the neighboring keyframes in the keyframe set         */
	/* Use these 3d points constructed by keyframes we found to estimate our camera pose(PnP) */

	cout << "Start matching scene with keyframes.\n";
	if (keyFrames.size() < 2)
	{
		cout << "KeyFrames size < 2\n";
		return false;
	}
	FindNeighboringKeyFrames(keyFrames, currData, neighboringKeyFrameIdx);
	if (neighboringKeyFrameIdx.size() == 0)
	{
		cout << "Neighboring keyframe size is zero.\n";
		return false;
	}
	for (std::vector<int>::iterator queryIdx = neighboringKeyFrameIdx.begin(); queryIdx != neighboringKeyFrameIdx.end(); ++queryIdx)
	{
		int r3dPtsCount = (int)keyFrames[*queryIdx].r3dPts.size();
		if (r3dPtsCount > 0)
		{
			//	Initialize the descriptors
			cv::Mat descriptors(r3dPtsCount, currData.descriptors.cols, currData.descriptors.type());
			for (int j = 0; j < r3dPtsCount; ++j)
				keyFrames[*queryIdx].descriptors.row(keyFrames[*queryIdx].coresIdx[j]).copyTo(descriptors.row(j));

			std::vector<cv::DMatch> matches, goodMatches;
			FlannMatching(currData.descriptors, descriptors, matches);
			FindGoodMatches(matches, goodMatches);
			goodMatchesSet.push_back(goodMatches);
		}
	}
	if (goodMatchesSet.size() == 0)	return false;

	cout << "Matching scene with keyframes is end.\n";

	return true;
}

/*	Triangulation	*/
void FeatureMatching(KeyFrame &query, KeyFrame &train, std::vector<cv::DMatch> &goodMatches)
{
	cout << "Start Triangulation matching.\n";
	std::vector<cv::DMatch> matches;
	FlannMatching(query.descriptors, train.descriptors, matches);
	FindGoodMatches(matches, goodMatches);
	cout << "Triangulation mathcing is end.\n";
}