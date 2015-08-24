#include <stdio.h>
#include <stdlib.h>
#include "DebugFunc.h"

void DebugSaveImage(char *filename, unsigned char *pbImage, int iWidth, int iHeight, int nChannel)
{
	FILE *fp;

	fp = fopen(filename,"wb");
	fwrite(pbImage,sizeof(unsigned char),iWidth*iHeight*nChannel,fp);
	fclose(fp);
}

void DebugLoadImage(char *filename, unsigned char *pbImage, int iWidth, int iHeight, int nChannel)
{
	FILE *fp;

	fp = fopen(filename,"rb");
	fread(pbImage,sizeof(unsigned char),iWidth*iHeight*nChannel,fp);
	fclose(fp);
}


void DrawPoint(unsigned char *Data, int Width, int Height, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	int xx,yy;

	Data[3*(y*Width+x)] = b;
	Data[3*(y*Width+x)+1] = g;
	Data[3*(y*Width+x)+2] = r;

	xx = x-1;
	yy = y;
	if(xx>=0 && xx<Width && yy>=0 && yy<Height)
	{
		Data[3*(yy*Width+xx)] = b;
		Data[3*(yy*Width+xx)+1] = g;
		Data[3*(yy*Width+xx)+2] = r;
	}

	xx = x+1;
	yy = y;
	if(xx>=0 && xx<Width && yy>=0 && yy<Height)
	{
		Data[3*(yy*Width+xx)] = b;
		Data[3*(yy*Width+xx)+1] = g;
		Data[3*(yy*Width+xx)+2] = r;
	}

	xx = x;
	yy = y-1;
	if(xx>=0 && xx<Width && yy>=0 && yy<Height)
	{
		Data[3*(yy*Width+xx)] = b;
		Data[3*(yy*Width+xx)+1] = g;
		Data[3*(yy*Width+xx)+2] = r;
	}

	xx = x;
	yy = y+1;
	if(xx>=0 && xx<Width && yy>=0 && yy<Height)
	{
		Data[3*(yy*Width+xx)] = b;
		Data[3*(yy*Width+xx)+1] = g;
		Data[3*(yy*Width+xx)+2] = r;
	}
}

void DebugMarkPoint(unsigned char* Data, int Width, int Height, vector<SiftGPU::SiftKeypoint> keys)
{
	int x,y;
	unsigned char r,g,b;

	srand(0);

	for (vector<SiftGPU::SiftKeypoint>::size_type i = 0; i<keys.size(); ++i)
	{
		r = rand()%256;
		g = rand()%256;
		b = rand()%256;

		x = (int)(keys[i].x+0.5);
		y = (int)(keys[i].y+0.5);

		DrawPoint(Data,Width,Height,x,y,r,g,b);
	}
}

void DebugMarkMatchedPoint(unsigned char* imgData, int Width, int Height, int nMatched,  int (*match_buf)[2], int ImageIndex, vector<SiftGPU::SiftKeypoint> keys)
{
	int x,y,i;
	unsigned char r,g,b;

	srand(0);

	for(i=0; i<nMatched; i++)
	{
		r = rand()%256;
		g = rand()%256;
		b = rand()%256;

		if(match_buf[i][ImageIndex] >= 0)
		{
			x = (int)(keys[match_buf[i][ImageIndex]].x+0.5); 
			y = (int)(keys[match_buf[i][ImageIndex]].y+0.5); 
			DrawPoint(imgData,Width,Height,x,y,r,g,b);
		}
	}
}

void DebugSaveCorrespondences(char *filename, vector<SiftGPU::SiftKeypoint> &keys1, vector<SiftGPU::SiftKeypoint> &keys2, int match_buf[][2], int num_match)
{
	FILE *fp;

	fp = fopen(filename,"wb");
	fwrite(&num_match,sizeof(int),1,fp);
	for(int i=0; i<num_match; ++i)
	{
		float x = keys1[match_buf[i][0]].x;
		float y = keys1[match_buf[i][0]].y;
		fwrite(&x,sizeof(float),1,fp);
		fwrite(&y,sizeof(float),1,fp);
		x = keys2[match_buf[i][0]].x;
		y = keys2[match_buf[i][0]].y;
		fwrite(&x,sizeof(float),1,fp);
		fwrite(&y,sizeof(float),1,fp);
	}
	fclose(fp);

	fp = fopen("Correspondences.txt","w");
	fprintf(fp,"%d\n",num_match);
	for(int i=0; i<num_match; ++i)
	{
		float x = keys1[match_buf[i][0]].x;
		float y = keys1[match_buf[i][0]].y;
		fprintf(fp,"%f %f ",x,y);
		x = keys2[match_buf[i][0]].x;
		y = keys2[match_buf[i][0]].y;
		fprintf(fp,"%f %f \n",x,y);
	}
	fclose(fp);
}

void DebugOpenCVMarkPoint(cv::Mat data, std::vector<cv::KeyPoint> keypoints, char *name)
{
	cv::Mat keypointsinimage;
	cv::drawKeypoints(data, keypoints, keypointsinimage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imshow(name, keypointsinimage);
	cv::imwrite(name, keypointsinimage);
	cv::waitKey(0);
}

void DebugOpenCVMatchPoint(cv::Mat image, std::vector<cv::KeyPoint> keypoints_img, cv::Mat scene, std::vector<cv::KeyPoint>keypoints_scene, std::vector<cv::DMatch>good_matches, char *name)
{
	cv::Mat matches;
	cv::drawMatches(image, keypoints_img, scene, keypoints_scene,
		good_matches, matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow(name, matches);
	cv::imwrite(name, matches);
	cv::waitKey(0);
}

void WriteVideo()
{
	cv::Mat curFrame;	//current frame
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
	int i = 0;

	cv::VideoWriter VW("GoProTestVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10.0, cv::Size(800, 600));


	while (cap.isOpened())
	{
		std::stringstream ss;
		ss << "TestRaw" << i << ".raw" << '\0';
		std::string s = ss.str();
		cap >> curFrame;
		cv::imshow("TestVideo", curFrame);
		if (cvWaitKey(20) == 27)
		{
			break;
		}
		VW << curFrame;
		char *c = new char[s.size()];
		strcpy(c, s.c_str());
		//DebugSaveImage(c, curFrame.data, 800, 600, 3);
		i++;
		delete[]c;
	}

}

void WriteRecordFile(unsigned long FrameCount, int GoodMatchesSize, int OpticalFlowKeyPointsSize)
{
	//std::ios::clear;
	std::fstream file;
	int GoodKeyPointsSize = GoodMatchesSize + OpticalFlowKeyPointsSize;
	if (GoodKeyPointsSize >= 30)
	{
		file.open("GoodMatchesSize.txt", std::ios::out | std::ios::app);
		file << "(FrameCount, GoodMatchesSize, OpticalFlowKeyPointsSize, GoodMatchesSize + OpticalFlowKeyPointsSize) : (" 
			 << FrameCount << ", " 
			 << (int)GoodMatchesSize << ", " 
			 << (int)OpticalFlowKeyPointsSize << ", "
			 << GoodKeyPointsSize
			 << ")\n";
	}
	else
	{
		file.open("GoodMatchesSize.txt", std::ios::out | std::ios::app);
		file << "NOTENOUGH (FrameCount, GoodMatchesSize, OpticalFlowKeyPointsSize, GoodMatchesSize + OpticalFlowKeyPointsSize) : ("
			 << FrameCount << ", " 
			 << (int)GoodMatchesSize << ", "
			 << (int)OpticalFlowKeyPointsSize << ", "
			 << GoodKeyPointsSize
			 << ")\n";
	}
	file.close();
}