#include <iostream>
#include <vector>
#include "Matrix\MyMatrix.h"
#include "BasicType.h"
#include "SFMUtil.h"
#include <math.h>
//#include "FundamentalMatrix\FundMatrix.h"
#include "PoseEstimation.h"
#include "FeatureProcess.h"
#include "MathLib\MathLib.h"
#include "levmar.h"

//For two views case
bool ReProjectToImage(MyMatrix &P, const cv::Point2f &pt, cv::Point3d &r3DPt)
{
	MyMatrix r3DptVec(4, 1);
	r3DptVec.m_lpdEntries[0] = r3DPt.x;
	r3DptVec.m_lpdEntries[1] = r3DPt.y;
	r3DptVec.m_lpdEntries[2] = r3DPt.z;
	r3DptVec.m_lpdEntries[3] = 1;

	MyMatrix homoReProjPt(3, 1);
	homoReProjPt = P * r3DptVec;

	cv::Point2f ordiReProjPt;
	ordiReProjPt.x = (float)homoReProjPt.m_lpdEntries[0] / homoReProjPt.m_lpdEntries[2];
	ordiReProjPt.y = (float)homoReProjPt.m_lpdEntries[1] / homoReProjPt.m_lpdEntries[2];


	float distance;
	distance = sqrtf((ordiReProjPt.x - pt.x)*(ordiReProjPt.x - pt.x) + (ordiReProjPt.y - pt.y)*(ordiReProjPt.y - pt.y));
	if (distance > 5.0f)
		return false;

	return true;
}

//For multiple views case
bool ReProjectToImage(std::vector<MyMatrix> &Ps, const std::vector<cv::Point2f> &pts, cv::Point3d &r3DPts)
{
	for (int i = 0; i < Ps.size(); ++i)
	{
		bool isCorrect = ReProjectToImage(Ps[i], pts[i], r3DPts);
		if (!isCorrect)
			return false;
	}

	return true;
}

/***************************************************************************************************************/
/*  Calculate 3D coordinates of points using known correspondence and projection matrices
/*  (See R.Hartley, "Multiple View Geometry in Computer Vision", 2nd Ed.page 312)
/*  Inpput: Ps: projection matrixs, pts: image correspondences in homogeneous coordinates
/*  Output: rPt: reconstructed 3D point in homogeneous coordinates
/***************************************************************************************************************/
bool Find3DCoordinates(const std::vector<MyMatrix> &Ps, const std::vector<Point3d> &pts, Point4d &r3DPt)
{
	int nViews;

	nViews = Ps.size();
	MyMatrix *A, *U, *D, *V;
	A = new MyMatrix(2 * nViews, 4);

	for (int i = 0; i<nViews; i++)
	{
		A->m_lpdEntries[2 * i * 4] = pts[i].x*Ps[i].m_lpdEntries[8] - pts[i].z*Ps[i].m_lpdEntries[0];
		A->m_lpdEntries[2 * i * 4 + 1] = pts[i].x*Ps[i].m_lpdEntries[9] - pts[i].z*Ps[i].m_lpdEntries[1];
		A->m_lpdEntries[2 * i * 4 + 2] = pts[i].x*Ps[i].m_lpdEntries[10] - pts[i].z*Ps[i].m_lpdEntries[2];
		A->m_lpdEntries[2 * i * 4 + 3] = pts[i].x*Ps[i].m_lpdEntries[11] - pts[i].z*Ps[i].m_lpdEntries[3];

		A->m_lpdEntries[(2 * i + 1) * 4] = pts[i].y*Ps[i].m_lpdEntries[8] - pts[i].z*Ps[i].m_lpdEntries[4];
		A->m_lpdEntries[(2 * i + 1) * 4 + 1] = pts[i].y*Ps[i].m_lpdEntries[9] - pts[i].z*Ps[i].m_lpdEntries[5];
		A->m_lpdEntries[(2 * i + 1) * 4 + 2] = pts[i].y*Ps[i].m_lpdEntries[10] - pts[i].z*Ps[i].m_lpdEntries[6];
		A->m_lpdEntries[(2 * i + 1) * 4 + 3] = pts[i].y*Ps[i].m_lpdEntries[11] - pts[i].z*Ps[i].m_lpdEntries[7];
	}
	U = new MyMatrix(2 * nViews, 2 * nViews);
	D = new MyMatrix(2 * nViews, 4);
	V = new MyMatrix(4, 4);

	A->SVD(U, D, V);

	r3DPt.x = V->m_lpdEntries[3];
	r3DPt.y = V->m_lpdEntries[7];
	r3DPt.z = V->m_lpdEntries[11];
	r3DPt.w = V->m_lpdEntries[15];


	delete A;
	delete U;
	delete D;
	delete V;

	return true;
}

/***************************************************************************************************************/
/*  Calculate 3D coordinates of points using known correspondence and projection matrices
/*  (See R.Hartley, "Multiple View Geometry in Computer Vision", 2nd Ed.page 312)
/*  Inpput: Ps: projection matrixs, pts: image correspondences in ordinary coordinates
/*  Output: rPt: reconstructed 3D point
/***************************************************************************************************************/
bool Find3DCoordinates(std::vector<MyMatrix> &Ps, const std::vector<cv::Point2f> &pts, cv::Point3d &r3DPt)
{
	int nViews;

	nViews = Ps.size();
	MyMatrix *A, *U, *D, *V;
	A = new MyMatrix(2 * nViews, 4);

	for (int i = 0; i<nViews; i++)
	{
		A->m_lpdEntries[2 * i * 4] = pts[i].x*Ps[i].m_lpdEntries[8] - Ps[i].m_lpdEntries[0];
		A->m_lpdEntries[2 * i * 4 + 1] = pts[i].x*Ps[i].m_lpdEntries[9] - Ps[i].m_lpdEntries[1];
		A->m_lpdEntries[2 * i * 4 + 2] = pts[i].x*Ps[i].m_lpdEntries[10] - Ps[i].m_lpdEntries[2];
		A->m_lpdEntries[2 * i * 4 + 3] = pts[i].x*Ps[i].m_lpdEntries[11] - Ps[i].m_lpdEntries[3];

		A->m_lpdEntries[(2 * i + 1) * 4] = pts[i].y*Ps[i].m_lpdEntries[8] - Ps[i].m_lpdEntries[4];
		A->m_lpdEntries[(2 * i + 1) * 4 + 1] = pts[i].y*Ps[i].m_lpdEntries[9] - Ps[i].m_lpdEntries[5];
		A->m_lpdEntries[(2 * i + 1) * 4 + 2] = pts[i].y*Ps[i].m_lpdEntries[10] - Ps[i].m_lpdEntries[6];
		A->m_lpdEntries[(2 * i + 1) * 4 + 3] = pts[i].y*Ps[i].m_lpdEntries[11] - Ps[i].m_lpdEntries[7];
	}
	U = new MyMatrix(2 * nViews, 2 * nViews);
	D = new MyMatrix(2 * nViews, 4);
	V = new MyMatrix(4, 4);

	A->SVD(U, D, V);

	if (V->m_lpdEntries[15] == 0.0)
	{
		delete A;
		delete U;
		delete D;
		delete V;
		return false;
	}
	else
	{
		r3DPt.x = V->m_lpdEntries[3] / V->m_lpdEntries[15];
		r3DPt.y = V->m_lpdEntries[7] / V->m_lpdEntries[15];
		r3DPt.z = V->m_lpdEntries[11] / V->m_lpdEntries[15];
		if (ReProjectToImage(Ps, pts, r3DPt))
		{
			delete A;
			delete U;
			delete D;
			delete V;
			return true;
		}
	}
	delete A;
	delete U;
	delete D;
	delete V;
	return false;
}

/***************************************************************************************************************/
/*  Calculate 3D coordinates of points using known correspondence and projection matrices
/*  (See R.Hartley, "Multiple View Geometry in Computer Vision", 2nd Ed.page 312)
/*  Inpput: Ps: projection matrixs, pts: image correspondences in ordinary coordinates
/*  Output: rPt: reconstructed 3D point
/***************************************************************************************************************/
bool Find3DCoordinates(MyMatrix &K, std::vector<MyMatrix> &Rs, std::vector<MyMatrix> &ts, const std::vector<Point2f> &pts, Point3d &r3DPt)
{
	//鄧老師改的
	int nViews;

	nViews = Rs.size();
	std::vector<MyMatrix> Ps;
	
	for(int i=0; i < nViews; i++)
	{
		MyMatrix P(3,4), P1(3,3),P2(3,1);
		
		//P1.Multiplication(&K, Rs);
		//K.Multiplication(&Rs, &P1);
		P1 = K*Rs[i];
		P2 = K*ts[i];
		
		P.m_lpdEntries[0] = P1.m_lpdEntries[0];
		P.m_lpdEntries[1] = P1.m_lpdEntries[1];
		P.m_lpdEntries[2] = P1.m_lpdEntries[2];
		P.m_lpdEntries[3] = P2.m_lpdEntries[0];
		
		P.m_lpdEntries[4] = P1.m_lpdEntries[3];
		P.m_lpdEntries[5] = P1.m_lpdEntries[4];
		P.m_lpdEntries[6] = P1.m_lpdEntries[5];
		P.m_lpdEntries[7] = P2.m_lpdEntries[1];
		
		P.m_lpdEntries[8] = P1.m_lpdEntries[6];
		P.m_lpdEntries[9] = P1.m_lpdEntries[7];
		P.m_lpdEntries[10] = P1.m_lpdEntries[8];
		P.m_lpdEntries[11] = P2.m_lpdEntries[2];
		Ps.push_back(P);
	}
//	Find3DCoordinates(Ps, pts, r3DPt);
	return true;
}

/***************************************************************************************************************/
/*  Calculate 3D coordinates of points using known correspondence and projection matrices (two-view case)
/*  (See R.Hartley, "Multiple View Geometry in Computer Vision", 2nd Ed.page 312)
/*  Inpput: Ps: projection matrixs, pts: image correspondences in ordinary coordinates
/*  Output: rPt: reconstructed 3D point
/***************************************************************************************************************/
bool Find3DCoordinates(MyMatrix &P1, MyMatrix &P2, const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Point3d &r3DPt)
{
	MyMatrix *A, *U, *D, *V;
	A = new MyMatrix(4, 4);

	A->m_lpdEntries[0] = pt1.x*P1.m_lpdEntries[8] - P1.m_lpdEntries[0];
	A->m_lpdEntries[1] = pt1.x*P1.m_lpdEntries[9] - P1.m_lpdEntries[1];
	A->m_lpdEntries[2] = pt1.x*P1.m_lpdEntries[10] - P1.m_lpdEntries[2];
	A->m_lpdEntries[3] = pt1.x*P1.m_lpdEntries[11] - P1.m_lpdEntries[3];
	A->m_lpdEntries[4] = pt1.y*P1.m_lpdEntries[8] - P1.m_lpdEntries[4];
	A->m_lpdEntries[5] = pt1.y*P1.m_lpdEntries[9] - P1.m_lpdEntries[5];
	A->m_lpdEntries[6] = pt1.y*P1.m_lpdEntries[10] - P1.m_lpdEntries[6];
	A->m_lpdEntries[7] = pt1.y*P1.m_lpdEntries[11] - P1.m_lpdEntries[7];

	A->m_lpdEntries[8] =  pt2.x*P2.m_lpdEntries[8] - P2.m_lpdEntries[0];
	A->m_lpdEntries[9] =  pt2.x*P2.m_lpdEntries[9] - P2.m_lpdEntries[1];
	A->m_lpdEntries[10] = pt2.x*P2.m_lpdEntries[10] - P2.m_lpdEntries[2];
	A->m_lpdEntries[11] = pt2.x*P2.m_lpdEntries[11] - P2.m_lpdEntries[3];
	A->m_lpdEntries[12] = pt2.y*P2.m_lpdEntries[8] - P2.m_lpdEntries[4];
	A->m_lpdEntries[13] = pt2.y*P2.m_lpdEntries[9] - P2.m_lpdEntries[5];
	A->m_lpdEntries[14] = pt2.y*P2.m_lpdEntries[10] - P2.m_lpdEntries[6];
	A->m_lpdEntries[15] = pt2.y*P2.m_lpdEntries[11] - P2.m_lpdEntries[7];

	U = new MyMatrix(4, 4);
	D = new MyMatrix(4, 4);
	V = new MyMatrix(4, 4);

	A->SVD(U, D, V);

	if (V->m_lpdEntries[15] == 0.0)
	{
		delete A;
		delete U;
		delete D;
		delete V;
		return false;
	}
	else
	{
		r3DPt.x = (float)V->m_lpdEntries[3] / V->m_lpdEntries[15];
		r3DPt.y = (float)V->m_lpdEntries[7] / V->m_lpdEntries[15];
		r3DPt.z = (float)V->m_lpdEntries[11] / V->m_lpdEntries[15];
		if (ReProjectToImage(P1, pt1, r3DPt) && ReProjectToImage(P2, pt2, r3DPt))
		{
			delete A;
			delete U;
			delete D;
			delete V;
			return true;
		}
		else
		{
			delete A;
			delete U;
			delete D;
			delete V;
			return false;
		}
	}
}

/***************************************************************************************************************/
/*  Calculate 3D coordinates of points using optimal triangulation method
/*  (See R.Hartley, "Multiple View Geometry in Computer Vision", page 318)
/*  This method can only be applied for two views
/*  Input: P1: projection matrix of view 1, P2: projection matrix of view 2, 
/*         pt1: image correspondences in view 1
/*         pt2: image correspondences in view 2 
/*         F: fundamental matrix of the two views 
/*  Output: rPt: reconstructed 3D point
/***************************************************************************************************************/
bool OptimalTriangulation(const MyMatrix &P1, const MyMatrix &P2, const cv::Point2f &pt1, const cv::Point2f &pt2, const MyMatrix &F, cv::Point3d &r3DPt)
{
	MyMatrix T1(3, 3), T2(3, 3);

	// T1 = [1 0 pt1.x; 0 1 pt1.y; 0 0 1]
	T1.m_lpdEntries[0] = 1.0; T1.m_lpdEntries[1] = 0.0; T1.m_lpdEntries[2] = pt1.x;
	T1.m_lpdEntries[3] = 0.0; T1.m_lpdEntries[4] = 1.0; T1.m_lpdEntries[5] = pt1.y;
	T1.m_lpdEntries[6] = 0.0; T1.m_lpdEntries[7] = 0.0; T1.m_lpdEntries[8] = 1.0;

	// T2 = [1 0 pt2.x; 0 1 pt2.y; 0 0 1]
	T2.m_lpdEntries[0] = 1.0; T2.m_lpdEntries[1] = 0.0; T2.m_lpdEntries[2] = pt2.x;
	T2.m_lpdEntries[3] = 0.0; T2.m_lpdEntries[4] = 1.0; T2.m_lpdEntries[5] = pt2.y;
	T2.m_lpdEntries[6] = 0.0; T2.m_lpdEntries[7] = 0.0; T2.m_lpdEntries[8] = 1.0;

	MyMatrix FF;// = T2.Transpose(&A)*F*T1;
	Point3d e1, e2;

	//if (!FindEpipole1(&FF, &e1)) return false;
	//if (!FindEpipole1(&FF.Transpose(), &e2)) return false;

	// Normalize the epipoles
	double len = sqrt(e1.x*e1.x + e1.y*e1.y);
	e1.x /= len; 	e1.y /= len;   e1.z /= len;

	len = sqrt(e2.x*e2.x + e2.y*e2.y);
	e2.x /= len; 	e2.y /= len;   e2.z /= len;

	MyMatrix R1(3, 3), R2(3, 3);

	// R1 = [e1.x e1.y 0; -e1.y e1.x 0; 0 0 1]
	R1.m_lpdEntries[0] =  e1.x; R1.m_lpdEntries[1] = e1.y; R1.m_lpdEntries[2] = 0.0;
	R1.m_lpdEntries[3] = -e1.y; R1.m_lpdEntries[4] = e1.x; R1.m_lpdEntries[5] = 0.0;
	R1.m_lpdEntries[6] = 0.0; R1.m_lpdEntries[7] = 0.0; R1.m_lpdEntries[8] = 1.0;

	// R2 = [e2.x e2.y 0; -e2.y e2.x 0; 0 0 1]
	R2.m_lpdEntries[0] =  e2.x; R2.m_lpdEntries[1] = e2.y; R2.m_lpdEntries[2] = 0.0;
	R2.m_lpdEntries[3] = -e2.y; R2.m_lpdEntries[4] = e2.x; R2.m_lpdEntries[5] = 0.0;
	R2.m_lpdEntries[6] = 0.0; R2.m_lpdEntries[7] = 0.0; R2.m_lpdEntries[8] = 1.0;

	MyMatrix RF = R2*FF*R1.Transpose();

	double f1 = e1.z; 
	double f2 = e2.z; 
	double a = RF.m_lpdEntries[4];  // RF(2,2)
	double b = RF.m_lpdEntries[5];  // RF(2,3)
	double c = RF.m_lpdEntries[7];  // RF(3,2)
	double d = RF.m_lpdEntries[8];  // RF(3,3)

	double coeff[7];

	coeff[0] = -(a*d - b*c)*a*c*f1*f1*f1*f1;
	coeff[1] = (a*a + f2*f2*c*c)*(a*a + f2*f2*c*c) - (a*d - b*c)*f1 * f1 *f1 *f1 * (b*c + a*d);
	coeff[2] = 4 * (a*a + f2*f2*c*c)*(a*b + f2*f2*c*d) - (a*d - b*c)*(2 * f1*f1*a*c + b*d*f1*f1*f1*f1);
	coeff[3] = 2 * (a*a + f2*f2*c*c)*(b*b + f2*f2*d*d) - 2 * (a*d - b*c)*f1*f1*(b*c + a*d);
	coeff[4] = 4 * (a*b + f2*f2*c*d)*(b*b + f2*f2*d*d) - (a*d - b*c)*(2 * f1*f1*b*d + a*c);
	coeff[5] = (b*b + f2*f2*d*d)* (b*b + f2*f2*d*d) - (a*d - b*c)*(b*c + a*d);
	coeff[6] = -(a*d - b*c)*b*d;

	double SolRealPart[6], SolImgPart[6];
	int nSol = PolynomialRoots(coeff, 6, SolRealPart, SolImgPart);
	double Min = 1E20, t_min = 0.0, cost;
	for (int i = 0; i < nSol; i++)
	{
		double t = SolRealPart[i];
		cost = t*t / (1 + f1*f1*t*t) + ((c*t + d) *(c*t + d)) / ((a*t + b)*(a*t + b) + f2*f2*(c*t + d)*(c*t + d));
		if (cost < Min)
		{
			Min = cost;
			t_min = t;
		}
	}

	double l1[3], l2[3];

	cost = 1 / (f1*f1) + c*c / (a*a + f2*f2*c*c);
	if (cost < Min)
	{
		l1[0] = f1; l1[1] = 0.0; l1[1] = -1.0;
		l2[0] = -f2*c; l2[1] = a; l2[2] = c;
	}
	else
	{
		l1[0] = t_min*f1; l1[1] = 1.0; l1[2] = -t_min;
		l2[0] = -f2*(c*t_min + d); l2[1] = a*t_min + b; l2[2] = c*t_min + d;
	}

	MyMatrix x1(3, 1), x2(3, 1);

	x1.m_lpdEntries[0] = -l1[0] * l1[2];
	x1.m_lpdEntries[1] = -l1[1] * l1[2];
	x1.m_lpdEntries[2] = l1[0] * l1[0] + l1[1] * l1[1];

	x2.m_lpdEntries[0] = -l2[0] * l2[2];
	x2.m_lpdEntries[1] = -l2[1] * l2[2];
	x2.m_lpdEntries[2] = l2[0] * l2[0] + l2[1] * l2[1];

	MyMatrix xx1 = T1*R1.Transpose()*x1;
	MyMatrix xx2 = T2*R2.Transpose()*x2;

	std::vector<MyMatrix> Ps;
	std::vector<cv::Point2f> pts;

	Ps.push_back(P1);
	Ps.push_back(P2);

	cv::Point2f pt;
	pt.x = xx1.m_lpdEntries[0] / xx1.m_lpdEntries[2];
	pt.y = xx1.m_lpdEntries[1] / xx1.m_lpdEntries[2];
	pts.push_back(pt);
	pt.x = xx2.m_lpdEntries[0] / xx2.m_lpdEntries[2];
	pt.y = xx2.m_lpdEntries[1] / xx2.m_lpdEntries[2];
	pts.push_back(pt);

	return Find3DCoordinates(Ps, pts, r3DPt);
}

/***************************************************************************************************************/
/*  Calculate camera parametres from Euclidean projection matrix (P=[K*R K*t])
/*  Input: P: 3 x 4 Projection matrix
/*  Output: K: 3 x 3 camera intrinsic parameters
/*          R: 3 x 3 camera rotation matrix
/*          t: 3 x 1 camera translation vector
/***************************************************************************************************************/
void CalculateCameraParameters(const MyMatrix &P, MyMatrix &K, MyMatrix &R, MyMatrix &t)
{
	MyMatrix B(3, 3);

	// B=P(1:3,1:3)
	B.m_lpdEntries[0] = P.m_lpdEntries[0]; B.m_lpdEntries[1] = P.m_lpdEntries[1]; B.m_lpdEntries[2] = P.m_lpdEntries[2];
	B.m_lpdEntries[3] = P.m_lpdEntries[4]; B.m_lpdEntries[4] = P.m_lpdEntries[5]; B.m_lpdEntries[5] = P.m_lpdEntries[6];
	B.m_lpdEntries[6] = P.m_lpdEntries[8]; B.m_lpdEntries[7] = P.m_lpdEntries[9]; B.m_lpdEntries[8] = P.m_lpdEntries[10];

	MyMatrix Q(3, 3), U(3, 3);

	B.Inverse().QRFactorization(Q, U);

	K = U.Inverse(); // K is a upper triangular matrix

	double s = K.m_lpdEntries[8];
	K.m_lpdEntries[0] /= s;
	K.m_lpdEntries[1] /= s;
	K.m_lpdEntries[2] /= s;
	K.m_lpdEntries[4] /= s;
	K.m_lpdEntries[5] /= s;
	K.m_lpdEntries[8] /= s;

	R = Q.Transpose();

	if (K.m_lpdEntries[0] < 0.0)
	{
		// K(:,1) = -K(:,1)
		K.m_lpdEntries[0] = -K.m_lpdEntries[0];
		K.m_lpdEntries[3] = -K.m_lpdEntries[3];
		K.m_lpdEntries[6] = -K.m_lpdEntries[6];
		// R(1,:) = -R(1,:)
		R.m_lpdEntries[0] = -R.m_lpdEntries[0];
		R.m_lpdEntries[1] = -R.m_lpdEntries[1];
		R.m_lpdEntries[2] = -R.m_lpdEntries[2];
	}
	if (K.m_lpdEntries[4] < 0.0)
	{
		// K(:,2) = -K(:,2)
		K.m_lpdEntries[1] = -K.m_lpdEntries[1];
		K.m_lpdEntries[4] = -K.m_lpdEntries[4];
		K.m_lpdEntries[7] = -K.m_lpdEntries[7];
		// R(2,:) = -R(2,:)
		R.m_lpdEntries[3] = -R.m_lpdEntries[3];
		R.m_lpdEntries[4] = -R.m_lpdEntries[4];
		R.m_lpdEntries[5] = -R.m_lpdEntries[5];
	}
	
	MyMatrix p4(3, 1);
	p4.m_lpdEntries[0] = P.m_lpdEntries[3];
	p4.m_lpdEntries[1] = P.m_lpdEntries[7];
	p4.m_lpdEntries[2] = P.m_lpdEntries[11];
	t = (K.Inverse()*p4)*(1/s);
}

/***************************************************************************************************************/
/*  Calculate camera focal length from fundamental matrix and principal point
/*  This approach is only suitable for two-view case and the fundamental matrix cannot be estimated 
/*  using correspondences from zero image principal point
/*  (See S. Bougnoux, “From projective to Euclidean space under any practical situation, a criticism of 
/*   self-calibration,” in Proceedings of the Sixth International Conference on Computer Vision
/*   (IEEE, 1998), pp. 790–796.
/*  Input: F: 3 x 3 Fundamental matrix 
/*         (u0, v0): image principal point for first view  
/*         (u1, v1): image principal point for second view
/*  Output: estimated camera focal length
/***************************************************************************************************************/
double EstimateFocalLength(MyMatrix &F, double u0, double v0, double u1, double v1)
{
	// F = 1000 * F / norm(F, 'fro');

	double norm = 0.0;
	for (int i = 0; i < F.m_iM*F.m_iN; ++i)
	{
		norm += F.m_lpdEntries[i] * F.m_lpdEntries[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < F.m_iM*F.m_iN; ++i)
	{
		F.m_lpdEntries[i] = 1000.0 * F.m_lpdEntries[i]/norm;
	}

	Point3d e;

	//FindEpipole1(&F, &e);

	//e = 1000 * e / norm(e);
	//e.normalize();
	e.x *= 1000;
	e.y *= 1000;
	e.z *= 1000;

	// E = [0 -e(3) e(2); e(3) 0 -e(1); -e(2) e(1) 0];
	MyMatrix E(3, 3);
	E.m_lpdEntries[0] = 0.0; E.m_lpdEntries[1] =-e.z; E.m_lpdEntries[2] = e.y;
	E.m_lpdEntries[3] = e.z; E.m_lpdEntries[4] = 0.0; E.m_lpdEntries[5] =-e.x;
	E.m_lpdEntries[6] =-e.y; E.m_lpdEntries[7] = e.x; E.m_lpdEntries[8] = 0.0;

	// I = [1 0 0;	0 1 0; 0 0 0];
	MyMatrix I(3, 3);
	I.m_lpdEntries[0] = 1.0; I.m_lpdEntries[1] = 0.0; I.m_lpdEntries[2] = 0.0;
	I.m_lpdEntries[3] = 0.0; I.m_lpdEntries[4] = 1.0; I.m_lpdEntries[5] = 0.0;
	I.m_lpdEntries[6] = 0.0; I.m_lpdEntries[7] = 0.0; I.m_lpdEntries[8] = 0.0;

	MyMatrix p1(3, 1), p2(3, 1);
	p1.m_lpdEntries[0] = u0; p1.m_lpdEntries[1] = v0; p1.m_lpdEntries[2] = 1.0;
	p2.m_lpdEntries[0] = u1; p2.m_lpdEntries[1] = v1; p2.m_lpdEntries[2] = 1.0;

	//f1 = p2'*E*I*F*p1*p1'*F'*p2;
	MyMatrix f1 = p2.Transpose()*E*I*F*p1*p1.Transpose()*F.Transpose()*p2;

	//f2 = p2'*E*I*F*I*F'*p2;
	MyMatrix f2 = p2.Transpose()*E*I*F*I*F.Transpose()*p2;

	return sqrt(-f1.m_lpdEntries[0] / f2.m_lpdEntries[0]);
}

void Triangulation(KeyFrame &KF1, KeyFrame &KF2, double *cameraPara)
{
	std::vector<cv::DMatch> matches;
	std::vector<cv::Point2f> KF1GoodMatches, KF2GoodMatches;
	std::vector<int> goodPts3DIdx;
	std::vector<cv::Point3d> pts3D;
	std::vector<cv::KeyPoint> keypointsMatches[2];
	cv::Mat descriptorsMatches[2];
	FlannMatching(KF1.descriptors, KF2.descriptors, matches);
	FindGoodMatches(KF1, KF2, matches, KF1GoodMatches, KF2GoodMatches, keypointsMatches, descriptorsMatches);
	if (KF1GoodMatches.size() == 0)
		return;

	/*MyMatrix K(3, 3);
	MyMatrix projMatrix1(3, 4), projMatrix2(3, 4);
	for (int i = 0; i < 9; ++i)
		K.m_lpdEntries[i] = cameraPara[i];
	CreateProjMatrix(K, KF1.R, KF1.t, projMatrix1);
	CreateProjMatrix(K, KF2.R, KF2.t, projMatrix2);*/

	for (int i = 0; i < KF1GoodMatches.size(); ++i)
	{
		//Triangulate
		cv::Point3d pt3D;
		if (Find3DCoordinates(KF1.projMatrix, KF2.projMatrix,
			KF1GoodMatches[i], KF2GoodMatches[i], pt3D))
		{
			pts3D.push_back(pt3D);
			goodPts3DIdx.push_back(i);
		}
	}
	for (int i = 0; i < pts3D.size(); ++i)
	{
		if ((pts3D[i].x > 400.0 || pts3D[i].x < -400.0
			|| pts3D[i].y > 300.0 || pts3D[i].y < -300.0))
		{
			KF1.r3dPts.push_back(pts3D[i]);
			KF2.r3dPts.push_back(pts3D[i]);
		}
		else
			goodPts3DIdx[i] = -1;
	}
	//initialize
	if (KF1.descriptors_3D.rows != 0 || KF1.descriptors_3D.cols != 0)
		KF1.descriptors_3D.create(KF1.r3dPts.size(), descriptorsMatches[0].cols, descriptorsMatches[0].type());
	if (KF2.descriptors_3D.rows != 0 || KF2.descriptors_3D.cols != 0)
		KF2.descriptors_3D.create(KF2.r3dPts.size(), descriptorsMatches[1].cols, descriptorsMatches[1].type());
	for (std::size_t i = 0, j = 0; i < goodPts3DIdx.size(); ++i)
	{
		if (goodPts3DIdx[i] != -1)
		{
			KF1.keypoints_3D.push_back(keypointsMatches[0][goodPts3DIdx[i]]);
			KF2.keypoints_3D.push_back(keypointsMatches[1][goodPts3DIdx[i]]);
			descriptorsMatches[0].row(goodPts3DIdx[i]).copyTo(KF1.descriptors_3D.row(j));
			descriptorsMatches[1].row(goodPts3DIdx[i]).copyTo(KF2.descriptors_3D.row(j));
			++j;
		}
	}

	std::vector<int>().swap(goodPts3DIdx);
	std::vector<cv::KeyPoint>().swap(keypointsMatches[0]);
	std::vector<cv::KeyPoint>().swap(keypointsMatches[1]);
	std::vector<cv::DMatch>  ().swap(matches);
	std::vector<cv::Point3d> ().swap(pts3D);
}

void Triangulation(double *cameraPara, std::vector<KeyFrame> &keyFrames, std::vector<int> &goodKeyFrameIdx)
{
	//std::vector<KeyFrame> matchedKFs;
	
	for (std::size_t i = 0; i < goodKeyFrameIdx.size(); ++i)
	{
		keyFrames[goodKeyFrameIdx[i]].r3dPts.clear();
		keyFrames[goodKeyFrameIdx[i]].keypoints_3D.clear();
		keyFrames[goodKeyFrameIdx[i]].descriptors_3D.release();
	}
	goodKeyFrameIdx.push_back(keyFrames.size() - 1);
	for (std::size_t i = 0; i < goodKeyFrameIdx.size(); ++i)
	{
		for (std::size_t j = i + 1; j < goodKeyFrameIdx.size(); ++j)
		{
			//std::vector<cv::DMatch> matches;
			//FlannMatching(keyFrames[goodKeyFrameIdx[i]].descriptors, keyFrames[goodKeyFrameIdx[j]].descriptors, matches);
		}
	}
}