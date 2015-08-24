#ifndef SFMUTIL_H
#define SFMUTIL_H

#include <vector>
#include "BasicType.h"
#include "Matrix\MyMatrix.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

bool Find3DCoordinates(const std::vector<MyMatrix> &Ps, const std::vector<Point3d> &pts, Point4d &r3DPt);
bool Find3DCoordinates(std::vector<MyMatrix> &Ps, const std::vector<cv::Point2f> &pts, cv::Point3f &r3DPt);
bool Find3DCoordinates(MyMatrix &P1, MyMatrix &P2, const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Point3f &r3DPt);
bool OptimalTriangulation(const MyMatrix &P1, const MyMatrix &P2, const Point2d &pt1, const Point2d &pt2, const MyMatrix &F, Point3d &r3DPt);
void CalculateCameraParameters(const MyMatrix &P, MyMatrix &K, MyMatrix &R, MyMatrix &t);
double EstimateFocalLength(MyMatrix &F, double u0, double v0, double u1, double v1);

#endif