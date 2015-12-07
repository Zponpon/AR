#include "FeatureProcess.h"

/*	Detection method : Surf(OpenCV)	*/
void SurfDetection(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian)
{
	int win = glutGetWindow();
	cv::SurfFeatureDetector detector(minHessian);
	detector.detect(image, keypoints);
	if (keypoints.size() == 0)
	{
		glutSetWindow(win);
		return;
	}
	cv::SurfDescriptorExtractor extractor;
	extractor.compute(image, keypoints, descriptors);

	glutSetWindow(win);
}

/*	Matching method : Flann(OpenCV)	*/
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

#pragma region FeatureDetection
bool FeatureDetection(FeatureMap &featureMap, unsigned int minHessian)
{
	//	Detect keypoints of featureMaps only
	SurfDetection(featureMap.image, featureMap.keypoints, featureMap.descriptors, minHessian);
	if (featureMap.keypoints.size() == 0)
		return false;
	return true;
}


bool FeatureDetection(FrameMetaData &currData, cv::Mat &currFrameMat, unsigned int minHessian )
{
	SurfDetection(currFrameMat, currData.keypoints, currData.descriptors, minHessian);
	if (currData.keypoints.size() == 0)
		return false;
	return true;
}
#pragma endregion

void RemoveDuplicatePts(std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches)
{
	//	把標記為True的點放入GoodMatchedPoints
	//	刪除OpticalFlow和Surf重複的特徵點

	std::vector<cv::Point2f> GoodMatchedPoints[2];	//[0]->featureMap(marker)中沒有重複的點, [1]->frame中沒有重複的點
	std::vector<bool> NotOverlayPointsFlag((unsigned int)featureMapGoodMatches.size(), true);	//標記keypoints x & y一樣的點為False

	for (int i = 0; i < (int)featureMapGoodMatches.size(); i++)
	{
		if (!NotOverlayPointsFlag[(std::vector<bool>::size_type)i])
			continue;
		for (int j = i + 1; j < (int)featureMapGoodMatches.size(); j++)
		{
			std::vector<cv::Point2f>::size_type indexI = (std::vector<cv::Point2f>::size_type) i;
			std::vector<cv::Point2f>::size_type indexJ = (std::vector<cv::Point2f>::size_type) j;
			float errX = std::abs(frameGoodMatches[indexI].x - frameGoodMatches[indexJ].x);
			float errY = std::abs(frameGoodMatches[indexI].y - frameGoodMatches[indexJ].y);
			if (!NotOverlayPointsFlag[(std::vector<bool>::size_type)j])
				continue;
			if (errX <= 5.0 && errY <= 5.0)	//誤差值該設定為多少??
				NotOverlayPointsFlag[(std::vector<bool>::size_type)j] = false;
		}
	}

	//位置接近的特徵點暫時先刪除OpticalFlow算出來的
	for (int i = 0; i < (int)NotOverlayPointsFlag.size(); i++)
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
	//Increase feature points
	//int GoodSize = 0;
	std::vector<cv::Point2f> GoodMatches[2]; //[0]->preFeatureMap, [1]->prevFrame, store the feature points
	std::vector<uchar> status;	//record OpticalPyrLK keypoints which are right tracking
	std::vector<float> error;	//record OpticalPyrLK error
	std::vector<cv::Point2f> OpticalFlow_keypoints;
	
	// OpenCV function to find OpticalFlow points
	cv::calcOpticalFlowPyrLK(prevFrame, currFrame, prevFrameGoodMatches, OpticalFlow_keypoints, status, error);

	for (int i = 0; i < (int)OpticalFlow_keypoints.size(); ++i)
	{
		//根據這個code來設定threshold https://github.com/Itseez/opencv/blob/master/samples/cpp/lkdemo.cpp/
		if (!(status[(std::vector<uchar>::size_type)i] == 0 || (cv::norm(OpticalFlow_keypoints[(std::vector<cv::Point2f>::size_type)i] - prevFeatureMapGoodMatches[(std::vector<cv::Point2f>::size_type)i]) <= 5)))	//目前為經驗法則
		{
			//避免拍到不是target image時，依舊保留上一個frame的特徵點
			GoodMatches[0].push_back(prevFeatureMapGoodMatches[(std::vector<cv::Point2f>::size_type)i]);
			GoodMatches[1].push_back(prevFrameGoodMatches[(std::vector<cv::Point2f>::size_type)i]);

			featureMapGoodMatches.push_back(prevFeatureMapGoodMatches[(std::vector<cv::Point2f>::size_type)i]);
			frameGoodMatches.push_back(OpticalFlow_keypoints[(std::vector<cv::Point2f>::size_type)i]);

			//GoodSize++;
		}
	}
	prevFeatureMapGoodMatches.swap(GoodMatches[0]);
	prevFrameGoodMatches.swap(GoodMatches[1]);
	//cout << "OpticalFlow Size : " << GoodSize << endl;
}

#pragma region FindGoodMatches
/*	Remove the false matching	*/
void FindGoodMatches(FeatureMap &featureMap, cv::Mat &currFrame, std::vector<cv::KeyPoint> &keypoints_frame, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches)
{
	//This method is to delete the feature point which is multimatch

	std::vector<cv::DMatch> goodMatches;
	//	Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//cout << "(max, min) : (" << max_dist << ", " << min_dist << ")\n\n";
	//	Find the good matches
	min_dist *= 2.0;
	if (min_dist > 0.2)
		min_dist = 0.2;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)matches[i].distance;
		if (dist < min_dist)
			goodMatches.push_back(matches[i]);
	}
	//標記一對多的點為false-> goodMatches
	std::vector<bool> GoodMatchesFlag(goodMatches.size(), true);	//判斷Surf算出來的點

	for (int j = 0; j < (int)goodMatches.size(); j++)
	{
		std::vector<cv::DMatch>::size_type index = (std::vector<cv::DMatch>::size_type) j;	//紀錄distance較小的index
		double distance = goodMatches[index].distance;

		if (!GoodMatchesFlag[(std::vector<bool>::size_type)index])	continue;

		for (int k = j + 1; k < (int)goodMatches.size(); k++)
		{
			if (!GoodMatchesFlag[(std::vector<bool>::size_type)k])	continue;

			if (goodMatches[index].trainIdx == goodMatches[(std::vector<cv::DMatch>::size_type)k].trainIdx)
			{
				if (distance <= goodMatches[(std::vector<cv::DMatch>::size_type)k].distance)
					GoodMatchesFlag[(std::vector<bool>::size_type)k] = false;
				else
				{
					GoodMatchesFlag[(std::vector<bool>::size_type)index] = false;
					index = (std::vector<cv::DMatch>::size_type)k;
					distance = goodMatches[index].distance;
				}
			}
		}
	}
	//DebugOpenCVMatchPoint(featureMap.image, featureMap.keypoints, frame.image, frame.keypoints, goodMatches, "IMG1.JPG");

	//把標記為true的點放入featureMap_goodMatches & frame_goodMatches
	for (int i = 0; i < (int)goodMatches.size(); i++)
	{
		if (GoodMatchesFlag[i])
		{
			featureMapGoodMatches.push_back(featureMap.keypoints[goodMatches[i].queryIdx].pt);
			frameGoodMatches.push_back(keypoints_frame[goodMatches[i].trainIdx].pt);
		}
	}
}

/*	Input : vector<Dmatch> (No optimization) */
/*	Ouput : vector<Dmatch> (optimization)	 */
void FindGoodMatches(std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &goodMatches)
{
	double max_dist = 0; double min_dist = 100;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//	Find the good matches
	min_dist *= 2.0;
	if (min_dist > 0.2) min_dist = 0.2;
	for (std::vector<cv::DMatch>::size_type i = 0; i < matches.size(); i++)
	{
		double dist = (double)matches[i].distance;
		if (dist < min_dist)
			goodMatches.push_back(matches[i]);
	}

	//標記一對多的點為false-> goodMatches
	std::vector<bool> GoodMatchesFlag((int)goodMatches.size(), true);
	for (int j = 0; j < (int)goodMatches.size(); j++)
	{
		std::vector<cv::DMatch>::size_type index = (std::vector<cv::DMatch>::size_type) j;	//紀錄distance較小的index
		double distance = goodMatches[index].distance;
		if (!GoodMatchesFlag[(std::vector<bool>::size_type)index])
			continue;
		for (int k = j + 1; k < (int)goodMatches.size(); k++)
		{
			if (!GoodMatchesFlag[k])
				continue;
			if (goodMatches[index].trainIdx == goodMatches[(std::vector<cv::DMatch>::size_type)k].trainIdx)
			{
				if (distance <= goodMatches[(std::vector<cv::DMatch>::size_type)k].distance)
					GoodMatchesFlag[k] = false;
				else
				{
					GoodMatchesFlag[(std::vector<bool>::size_type)index] = false;
					index = (std::vector<cv::DMatch>::size_type)k;
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
#pragma endregion

#pragma region FeatureMatch
bool FeatureMatching(FeatureMap &featureMap, FrameMetaData &currData, cv::Mat &currFrameMat, cv::Mat &prevFrameMat, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers)
{
	//Matching current scene with the featureMap

	std::vector<cv::DMatch> matches;
	FlannMatching(featureMap.descriptors, currData.descriptors, matches);
	FindGoodMatches(featureMap, currFrameMat, currData.keypoints, matches, featureMapGoodMatches, currFrameGoodMatches);

	//If GoodMatches size is not enough, use OpticalFlow to increase more matching points
	if ((int)currFrameGoodMatches.size() < 25)
	{
		if ((int)prevFrameInliers.size() != 0 && (int)prevFeatureMapInliers.size() != 0 && prevFrameMat.data != NULL)
		{
			//Use optical flow to detect matching points
			OpticalFlow(prevFrameMat, currFrameMat, prevFrameInliers, prevFeatureMapInliers, featureMapGoodMatches, currFrameGoodMatches);
			RemoveDuplicatePts(featureMapGoodMatches, currFrameGoodMatches);
			if ((int)currFrameGoodMatches.size() < 25)
				return false;
		}
		else
		{
			prevFeatureMapInliers.swap(featureMapGoodMatches);
			prevFrameInliers.swap(currFrameGoodMatches);
			return false;
		}
	}

	return true;
}


/*	Ransac PnP	*/
bool FeatureMatching(std::vector<SFM_Feature> &SFM_Features, std::vector<KeyFrame> &keyFrames, FrameMetaData &currData, cv::Mat &currFrameMat, std::vector<int> &neighboringKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet)
{
	if (keyFrames.size() < 2)
		return false;

	cout << "Start matching scene with keyframes.\n";

	//FindNeighboringKeyFrames(keyFrames, currData, neighboringKeyFrameIdx);
	if (neighboringKeyFrameIdx.size() == 0)
	{
		cout << "Neighboring keyframe size is zero.\n";
		return false;
	}

	for (std::vector<int>::iterator queryIdx = neighboringKeyFrameIdx.begin(); queryIdx != neighboringKeyFrameIdx.end(); ++queryIdx)
	{
		int r3dPtsCount = (int)keyFrames[*queryIdx].ptIdx.size();
		if (r3dPtsCount == 0)	continue;

		int index = 0;
		cv::Mat descriptors(r3dPtsCount, currData.descriptors.cols, CV_32F);
		std::vector<cv::KeyPoint> pts;// For debug

		for (std::vector<SFM_Feature>::iterator feature = SFM_Features.begin(); feature != SFM_Features.end(); ++feature)
		{
			if (feature->imgIdx == *queryIdx && feature->ptIdx != -1)
			{
				keyFrames[*queryIdx].descriptors.row(feature->descriptorIdx).copyTo(descriptors.row(index)); 
				pts.push_back(keyFrames[*queryIdx].keypoints[feature->descriptorIdx]);
				index++;
			}
		}

		//	keyframe->query, current frame->train
		std::vector<cv::DMatch> matches, goodMatches;
		
		FlannMatching(descriptors, currData.descriptors, matches);

		FindGoodMatches(matches, goodMatches);

	//	DebugOpenCVMatchPoint(keyFrames[*queryIdx].image, pts, currFrameImg, currData.keypoints, goodMatches, "PnPMatching.jpg");

		if (goodMatches.size() == 0) continue;

		goodMatchesSet.push_back(goodMatches);
	}

	if (goodMatchesSet.size() == 0)	
		return false;

	cout << "Matching scene with keyframes is end.\n";

	return true;
}


/*	Triangulation	*/
void FeatureMatching(KeyFrame &query, KeyFrame &train, std::vector<cv::DMatch> &goodMatches)
{
	cout << "Start Triangulation matching.\n";

	std::vector<cv::DMatch> matches;

	FlannMatching(query.descriptors, train.descriptors, matches);

	//DebugOpenCVMatchPoint(query.image, query.keypoints, train.image, train.keypoints, matches, "TestNotGood.jpg");

	FindGoodMatches(matches, goodMatches);

	DebugOpenCVMatchPoint(query.image, query.keypoints, train.image, train.keypoints, goodMatches, "TriangulationGoodMatching.jpg");
	
	cout << "Triangulation mathcing is end.\n";
}
#pragma endregion