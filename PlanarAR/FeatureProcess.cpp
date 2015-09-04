#include "FeatureProcess.h"

void FindGoodMatches(FeatureMap &featureMap, Frame &frame, std::vector<cv::KeyPoint> &keypoints_frame, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &featureMap_goodMatches, std::vector<cv::Point2f> &frame_goodMatches)
{
	//This method is to delete the feature points which are multimatch
	
	std::vector<cv::DMatch> goodMatches;
	//	Quick calcul ation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for (std::size_t i = 0; i < matches.size(); i++)
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
	for (std::size_t i = 0; i < matches.size(); i++)
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
			featureMap_goodMatches.push_back(featureMap.keypoints[goodMatches[i].queryIdx].pt);
			frame_goodMatches.push_back(keypoints_frame[goodMatches[i].trainIdx].pt);
		}
	}

	//Release memory
	std::vector<cv::DMatch>().swap(goodMatches);
	std::vector<bool>().swap(GoodMatchesFlag);
}

void FindGoodMatches(Frame &img1, Frame &img2, std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &img1_goodMatches, std::vector<cv::Point2f> &img2_goodMatches, std::vector<cv::KeyPoint> &keypoints_3D, cv::Mat &descriptors_3D)
{
	//This method is to triangulate points, and swap good matching keypoints & descriptors

	//	Quick calcul ation of max and min distances between keypoints
	std::vector<cv::DMatch> goodMatches;
	double max_dist = 0; double min_dist = 100;
	for (std::size_t i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//	Find the good matches
	min_dist *= 2.0;
	if (min_dist > 0.2) min_dist = 0.2;
	for (std::size_t i = 0; i < matches.size(); i++)
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
	//把標記為true的點放入featureMap_goodMatches & frame_goodMatches
	cv::Mat tempdescriptors(goodMatches.size(), img2.descriptors.cols, img2.descriptors.type());
	std::vector<cv::KeyPoint> tempkeypoints;
	std::vector<cv::DMatch> tempMatch;
	for (std::size_t i = 0, j = 0; i < goodMatches.size(); ++i)
	{
		if (GoodMatchesFlag[i])
		{
			//tempMatch.push_back(goodMatches[i]);
			img1_goodMatches.push_back(img1.keypoints[goodMatches[i].queryIdx].pt);
			img2_goodMatches.push_back(img2.keypoints[goodMatches[i].trainIdx].pt);
			tempkeypoints.push_back(img1.keypoints[goodMatches[i].queryIdx]);
			keypoints_3D.push_back(img2.keypoints[goodMatches[i].trainIdx]);
			tempMatch.push_back(goodMatches[i]);
			tempMatch[j].queryIdx = j;
			tempMatch[j].trainIdx = j;
			img2.descriptors.row(goodMatches[i].trainIdx).copyTo(tempdescriptors.row(j));//save the match points's descriptors
			++j;
		}
	}

	descriptors_3D.create(keypoints_3D.size(), img2.descriptors.cols, img2.descriptors.type());
	for (std::size_t i = 0; i < keypoints_3D.size(); ++i)
		tempdescriptors.row(i).copyTo(descriptors_3D.row(i));
#ifdef SHOWTHEIMAGE
	DebugOpenCVMatchPoint(img1.image, tempkeypoints, img2.image, keypoints_3D, tempMatch, "APoints.jpg");
#endif
	//free vector memory
	std::vector<cv::KeyPoint>().swap(tempkeypoints);
	std::vector<cv::DMatch>().swap(matches);
	std::vector<cv::DMatch>().swap(goodMatches);
	std::vector<bool>().swap(GoodMatchesFlag);
}

void FindGoodMatches(Frame &prevImage, Frame &newImage, std::vector<cv::DMatch> &matches, std::vector<int> &good3DMatches, std::vector<cv::Point2f> &newImage_goodMatches)
{
	//This method is to find 3D points which are good match

	std::vector<cv::DMatch> goodMatches;

	double max_dist = 0; double min_dist = 100;
	for (std::size_t i = 0; i < matches.size(); ++i)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//	Find the good matches
 	min_dist *= 2.0;
	if (min_dist > 0.2)
		min_dist = 0.2;
	for (std::size_t i = 0; i < matches.size(); i++)
	{
		double dist = (double)(matches[i].distance);
		if (dist < min_dist)
			goodMatches.push_back(matches[i]);
	}

	//標記一對多的點為false-> goodMatches
	std::vector<bool> GoodMatchesFlag(goodMatches.size(), true);	//判斷Surf算出來的點

	for (std::size_t j = 0; j < goodMatches.size(); ++j)
	{
		std::size_t index = j;	//紀錄distance較小的index
		double distance = goodMatches[index].distance;
		if (!GoodMatchesFlag[index]) continue;
		for (std::size_t k = j + 1; k < goodMatches.size(); ++k)
		{
			if (!GoodMatchesFlag[k]) continue;
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
	std::vector<cv::DMatch> temp;
	for (std::size_t i = 0; i < goodMatches.size(); ++i)
	{
		if (GoodMatchesFlag[i])
		{
			temp.push_back(goodMatches[i]);
			good3DMatches.push_back(goodMatches[i].queryIdx);//Query the features index from the previous frame
			newImage_goodMatches.push_back(newImage.keypoints[goodMatches[i].trainIdx].pt);
		}
	}
	DebugOpenCVMarkPoint(prevImage.image, prevImage.keypoints_3D, "KEYPOINT1.JPG");
	DebugOpenCVMarkPoint(newImage.image, newImage.keypoints, "KEYPOINT2.JPG");
	DebugOpenCVMatchPoint(prevImage.image, prevImage.keypoints_3D, newImage.image, newImage.keypoints, temp, "VisualOdometry.JPG");
	std::vector<cv::DMatch>().swap(goodMatches);
	std::vector<bool>().swap(GoodMatchesFlag);
}

void DeleteOverlap(std::vector<cv::Point2f> &featureMap_goodMatches, std::vector<cv::Point2f> &frame_goodMatches)
{
	//	把標記為True的點放入GoodMatchedPoints
	//	刪除OpticalFlow和Surf重複的特徵點

	std::vector<cv::Point2f> GoodMatchedPoints[2];	//[0]->featureMap(marker)中沒有重複的點, [1]->frame中沒有重複的點
	std::vector<bool> NotOverlayPointsFlag(featureMap_goodMatches.size(), true);	//標記keypoints x & y一樣的點為False

	for (std::size_t i = 0; i < featureMap_goodMatches.size(); i++)
	{
		if (!NotOverlayPointsFlag[i])
			continue;
		for (std::size_t j = i + 1; j < featureMap_goodMatches.size(); j++)
		{
			float errX = std::abs(frame_goodMatches[i].x - frame_goodMatches[j].x);
			float errY = std::abs(frame_goodMatches[i].y - frame_goodMatches[j].y);
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
			GoodMatchedPoints[0].push_back(featureMap_goodMatches[i]);
			GoodMatchedPoints[1].push_back(frame_goodMatches[i]);
		}
	}
	featureMap_goodMatches.swap(GoodMatchedPoints[0]);
	frame_goodMatches.swap(GoodMatchedPoints[1]);

	//free vector memory
	std::vector<cv::Point2f>().swap(GoodMatchedPoints[0]);
	std::vector<cv::Point2f>().swap(GoodMatchedPoints[1]);
	std::vector<bool>().swap(NotOverlayPointsFlag);
}

void SurfDetection(cv::Mat &data, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian)
{
	int win = glutGetWindow();
	cv::SurfFeatureDetector *detector = new cv::SurfFeatureDetector(minHessian);
	detector->detect(data, keypoints);
	if (keypoints.size() == 0){
		glutSetWindow(win);
		return;
	}
	cv::SurfDescriptorExtractor *extractor = new cv::SurfDescriptorExtractor;
	extractor->compute(data, keypoints, descriptors);

	delete detector;
	delete extractor;
	glutSetWindow(win);
}

void FlannMatching(cv::Mat &descriptors_featureMap, cv::Mat &descriptors_frame, std::vector<cv::DMatch> &matches)
{
	int win = glutGetWindow();
	cv::FlannBasedMatcher *matcher = new cv::FlannBasedMatcher();
	if (descriptors_featureMap.type() != CV_32F)
		descriptors_featureMap.convertTo(descriptors_featureMap, CV_32F);
	if (descriptors_frame.type() != CV_32F)
		descriptors_frame.convertTo(descriptors_frame, CV_32F);
	matcher->match(descriptors_featureMap, descriptors_frame, matches);
	delete matcher;
	glutSetWindow(win);
}

void OpticalFlow(FeatureMap &featureMap, Frame &prevFrame, Frame &currFrame, std::vector<cv::Point2f> &currPts, std::vector<int> &good3DMatches)
{
	std::vector<cv::Point2f> prevPts;
	std::vector<cv::Point2f> tempPts;
	//cv::KeyPoint::convert(prevFrame.keypoints_3D, prevPts);
	std::vector<cv::Point2f> ReprojPts;
	cv::KeyPoint::convert(featureMap.reProjPts, ReprojPts);

	std::vector<uchar> status;
	std::vector<float> error;
	cv::calcOpticalFlowPyrLK(prevFrame.image, currFrame.image, ReprojPts, tempPts, status, error);
	for (int i = 0; i < tempPts.size(); ++i)
	{
		if (!(status[i] == 0 || error[i] > 5))
		{
			currPts.push_back(tempPts[i]);
			good3DMatches.push_back(i);
		}
	}
	std::vector<cv::Point2f>().swap(ReprojPts);
	//std::vector<cv::Point2f>().swap(prevPts);
	std::vector<cv::Point2f>().swap(tempPts);
	std::vector<uchar>().swap(status);
	std::vector<float>().swap(error);
}

void OpticalFlow(cv::Mat prevFrame, cv::Mat &currFrame, std::vector<cv::Point2f> &prevFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &featureMap_goodMatches, std::vector<cv::Point2f> &frame_goodMatches)
{
	//For PoseEsitmation by homography
	int GoodSize = 0;
	std::vector<cv::Point2f> GoodMatches[2]; //[0]->preFeatureMap, [1]->prevFrame, store the feature points
	std::vector<uchar> status;	//record OpticalPyrLK keypoints which are right tracking
	std::vector<float> error;	//record OpticalPyrLK error
	std::vector<cv::Point2f> OpticalFlow_keypoints;
	
	// OpenCV function to find OpticalFlow points
	cv::calcOpticalFlowPyrLK(prevFrame, currFrame, prevFrameGoodMatches, OpticalFlow_keypoints, status, error);

	for (std::size_t i = 0; i < OpticalFlow_keypoints.size(); ++i)
	{
		//兩張圖error門檻值取大於30都不錯
		//error似乎只是判斷有無值可以用(cv::norm(OpticalFlow_keypoints[i] - prevFeatureMapGoodMatches[i]) < 5))
		if (!(status[i] == 0 || (cv::norm(OpticalFlow_keypoints[i] - prevFeatureMapGoodMatches[i]) < 5)))	//目前為經驗法則
		{
			GoodMatches[0].push_back(prevFeatureMapGoodMatches[i]);//避免拍到不是target image時，依舊保留上一個frame的特徵點
			GoodMatches[1].push_back(prevFrameGoodMatches[i]);
			featureMap_goodMatches.push_back(prevFeatureMapGoodMatches[i]);
			frame_goodMatches.push_back(OpticalFlow_keypoints[i]);
			GoodSize++;
		}
	}
	prevFeatureMapGoodMatches.swap(GoodMatches[0]);
	prevFrameGoodMatches.swap(GoodMatches[1]);

	//因為他們的Size都會一樣，所以用prevFrameGoodMatches的Size表示就好
	std::cout << "OpticalFlow Size : " << GoodSize << std::endl;

	//	free vector memory
	std::vector<cv::Point2f>().swap(GoodMatches[0]);
	std::vector<cv::Point2f>().swap(GoodMatches[1]);
	std::vector<cv::Point2f>().swap(OpticalFlow_keypoints);
	std::vector<uchar>().swap(status);
	std::vector<float>().swap(error);
}

bool FeatureDetectionAndMatching(FeatureMap &featureMap, Frame &prevFrame, Frame &currFrame, unsigned int minHessian, std::vector<cv::Point2f>  &featureMap_goodMatches, std::vector<cv::Point2f> &frame_goodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &prevFrameGoodMatches)
{
	//For homography pose estimation
	SurfDetection(currFrame.image, currFrame.keypoints, currFrame.descriptors, minHessian);
	if (currFrame.keypoints.size() == 0) return false;
	std::vector<cv::DMatch> matches;
	FlannMatching(featureMap.descriptors, currFrame.descriptors, matches);
	FindGoodMatches(featureMap, currFrame, currFrame.keypoints, matches, featureMap_goodMatches, frame_goodMatches);
	std::cout << "Surf Good Matches Size : " << frame_goodMatches.size() << std::endl;

	bool usingOpticalFlow = false;
	/*if ((int)frame_goodMatches.size() < 5)
	{
		prevFeatureMapGoodMatches.swap(featureMap_goodMatches);
		prevFrameGoodMatches.swap(frame_goodMatches);
		return false;
	}*/
	if ((int)frame_goodMatches.size() < 25)
	{
		if (prevFrameGoodMatches.size() != 0 && prevFeatureMapGoodMatches.size() != 0 && prevFrame.image.data != NULL)
		{
			OpticalFlow(prevFrame.image, currFrame.image, prevFrameGoodMatches, prevFeatureMapGoodMatches, featureMap_goodMatches, frame_goodMatches);
			usingOpticalFlow = true;	//Surf偵測的特徵點不足
		}
		else
		{
			prevFeatureMapGoodMatches.swap(featureMap_goodMatches);
			prevFrameGoodMatches.swap(frame_goodMatches);

			std::vector<cv::DMatch>().swap(matches);
			std::vector<cv::KeyPoint>().swap(currFrame.keypoints);
			std::vector<cv::Point2f>().swap(featureMap_goodMatches);
			std::vector<cv::Point2f>().swap(frame_goodMatches);

			return false;
		}
	}

	if (usingOpticalFlow)
	{
		DeleteOverlap(featureMap_goodMatches, frame_goodMatches);
		std::cout << "Surf + OpticalFlow Good Matches Size : " << frame_goodMatches.size() << std::endl;
		if ((int)frame_goodMatches.size() < 30)
		{
			std::vector<cv::DMatch>().swap(matches);
			std::vector<cv::KeyPoint>().swap(currFrame.keypoints);
			std::vector<cv::Point2f>().swap(featureMap_goodMatches);
			std::vector<cv::Point2f>().swap(frame_goodMatches);

			return false;
		}
	}
	std::vector<cv::DMatch>().swap(matches);

	return true;
}
