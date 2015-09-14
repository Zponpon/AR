#include <windows.h>
#include <iostream>
#include <GL/gl.h>
#include <map>
#include <cmath>
#include "glut.h"
#include "BasicType.h"
#include "Matrix\MyMatrix.h"
#include "Homography\Homography.h"
#include "SiftGPU.h"
#include "PoseEstimation.h"
#include "SFMUtil.h"
#include "FeatureProcess.h"
#include "levmar.h"
#include "debugfunc.h"

//std::vector<cv::Point2f> prevFrameinliers; //Point2f vector good keypoints in previous frame
//std::vector<cv::Point2f> prevFeatureMapinliers; //Point2f vector good keypoints in image

/*********************************************************************************************************/
/* Given rotation axis and rotation angle, find the corresponding rotation matrix                        */
/* Input: angle: rotation angle, axis: rotation axis                                                     */
/* Output: R: rotation matrix                                                                            */
/*********************************************************************************************************/
void RotationMatrix(double angle, double axis[3], MyMatrix &R)
{
	double c, s, len;

	c = cos(angle);
	s = sin(angle);

	len = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

	double u=axis[0]/len;
	double v=axis[1]/len;
	double w=axis[2]/len;

	R.m_lpdEntries[0] = u*u*(1-c)+c; R.m_lpdEntries[1] = u*v*(1-c)-w*s;  R.m_lpdEntries[2] = u*w*(1-c)+v*s;
	R.m_lpdEntries[3] = u*v*(1-c)+w*s; R.m_lpdEntries[4] = v*v*(1-c)+c;  R.m_lpdEntries[5] = v*w*(1-c)-u*s;
	R.m_lpdEntries[6] = u*w*(1-c)-v*s; R.m_lpdEntries[7] = v*w*(1-c)+u*s;  R.m_lpdEntries[8] = w*w*(1-c)+c;
}

/*********************************************************************************************************/
/* Given rotation matrix, find the corresponding rotation angle and axis                                 */
/* Input: R: rotation matrix                                                                             */
/* Output: angle: rotation angle, axis: rotation axis                                                    */
/*********************************************************************************************************/
void GetRotationAxisAndAngle(MyMatrix &R, double &angle, double axis[3])
{
	angle = acos(0.5*(R.m_lpdEntries[0]+R.m_lpdEntries[4]+R.m_lpdEntries[8]-1)); // trace(R)=2cos(angle)+1

	MyMatrix RI(3,3);

	// RI = R-I
	RI.m_lpdEntries[0] = R.m_lpdEntries[0]-1.0;
	RI.m_lpdEntries[1] = R.m_lpdEntries[1];
	RI.m_lpdEntries[2] = R.m_lpdEntries[2];

	RI.m_lpdEntries[3] = R.m_lpdEntries[3];
	RI.m_lpdEntries[4] = R.m_lpdEntries[4]-1.0;
	RI.m_lpdEntries[5] = R.m_lpdEntries[5];

	RI.m_lpdEntries[6] = R.m_lpdEntries[6];
	RI.m_lpdEntries[7] = R.m_lpdEntries[7];
	RI.m_lpdEntries[8] = R.m_lpdEntries[8]-1.0;

	MyMatrix U(3,3),S(3,3),V(3,3);

	RI.SVD(&U,&S,&V);

	axis[0] = V.m_lpdEntries[2];
	axis[1] = V.m_lpdEntries[5];
	axis[2] = V.m_lpdEntries[8];

	MyMatrix newR(3,3);

	RotationMatrix(angle,axis,newR);

	double err=0.0;

	for(int i=0; i<R.m_iM*R.m_iN; i++)
	{
		err += (R.m_lpdEntries[i]-newR.m_lpdEntries[i])*(R.m_lpdEntries[i]-newR.m_lpdEntries[i]);
	}

	if(err > 0.001)
	{
		axis[0] = -axis[0];
		axis[1] = -axis[1];
		axis[2] = -axis[2];
	}
}

/*******************************************************************************************************/
/* Given camera intrinsic matrix, estimate camera pose from homography without considering             */   
/* the orthonormality of r1 and r2                                                                     */
/* Because H = K[r1 r2 t] => [r1 r2 t] = inv(K)*H                                                      */
/* Input: H: homography matrix, K:camera intrinsic matrix                                              */
/* Output: t: camera translation, R=[r1 r2 r3]: camera rotation                                        */
/*******************************************************************************************************/
void EstimateCameraPoseFromHomographyLite(MyMatrix &H, MyMatrix &K, MyMatrix &R, Vector3d &t)
{
	MyMatrix invK(3,3);
	MyMatrix invKH(3,3);

	K.Inverse(&invK);

	invK.Multiplication(&H,&R); // R = inv(k)*H

	double norm1 = sqrt(R.m_lpdEntries[0]*R.m_lpdEntries[0] + R.m_lpdEntries[3]*R.m_lpdEntries[3] + R.m_lpdEntries[6]*R.m_lpdEntries[6]);
	double norm2 = sqrt(R.m_lpdEntries[1]*R.m_lpdEntries[1] + R.m_lpdEntries[4]*R.m_lpdEntries[4] + R.m_lpdEntries[7]*R.m_lpdEntries[7]);
	double scale;

	// Determine an appropriate scale from the norm of column vectors of R
	if(norm1 > norm2) scale = norm1;
	else scale = norm2;

	if(R.m_lpdEntries[8]<0.0) scale=-scale; // Impose the constraint that the z coordinate of translation (R(3,3)) is greater than 0 (positive z-axis)

	// Scale the rotation matrix
	for(int i=0; i<9;++i) R.m_lpdEntries[i]/=scale;

	t.x = R.m_lpdEntries[2];
	t.y = R.m_lpdEntries[5];
	t.z = R.m_lpdEntries[8];

	// r3 = r1 x r2
	R.m_lpdEntries[2] = R.m_lpdEntries[3] * R.m_lpdEntries[7] - R.m_lpdEntries[6] * R.m_lpdEntries[4];
	R.m_lpdEntries[5] = R.m_lpdEntries[6] * R.m_lpdEntries[1] - R.m_lpdEntries[0] * R.m_lpdEntries[7];
	R.m_lpdEntries[8] = R.m_lpdEntries[0] * R.m_lpdEntries[4] - R.m_lpdEntries[3] * R.m_lpdEntries[1];
}

/*******************************************************************************************************/
/* Given camera intrinsic matrix, estimate camera pose from homography with considering                */   
/* the orthonormality of r1 and r2                                                                     */
/* Because H = K[r1 r2 t] => [r1 r2 t] = inv(K)*H                                                      */
/* Input: H: homography matrix, K:camera intrinsic matrix                                              */
/* Output: t: camera translation, R=[r1 r2 r3]: camera rotation                                        */
/*******************************************************************************************************/
void EstimateCameraPoseFromHomography(MyMatrix &H, MyMatrix &K, MyMatrix &R, Vector3d &t)
{
	MyMatrix invK(3,3);
	MyMatrix invKH(3,3);

	K.Inverse(&invK);

	invK.Multiplication(&H, &R); // R = inv(k)*H

	double norm1 = sqrt(R.m_lpdEntries[0]*R.m_lpdEntries[0] + R.m_lpdEntries[3]*R.m_lpdEntries[3] + R.m_lpdEntries[6]*R.m_lpdEntries[6]);
	double norm2 = sqrt(R.m_lpdEntries[1]*R.m_lpdEntries[1] + R.m_lpdEntries[4]*R.m_lpdEntries[4] + R.m_lpdEntries[7]*R.m_lpdEntries[7]);
	double scale;

	// Determine an appropriate scale from the norm of column vectors of R
	if(norm1 > norm2) scale = norm1;
	else scale = norm2;

	if(R.m_lpdEntries[8]<0.0) scale=-scale; // Impose the constraint that the z coordinate of translation (R(3,3)) is greater than 0 (positive z-axis)

	// Scale the rotation matrix
	for(int i=0; i<9;++i) R.m_lpdEntries[i]/=scale;

	t.x = R.m_lpdEntries[2];
	t.y = R.m_lpdEntries[5];
	t.z = R.m_lpdEntries[8];

	double r1[3],r2[3],r3[3];

	// re-normalize r1 and r2
	norm1 = sqrt(R.m_lpdEntries[0]*R.m_lpdEntries[0] + R.m_lpdEntries[3]*R.m_lpdEntries[3] + R.m_lpdEntries[6]*R.m_lpdEntries[6]);
	norm2 = sqrt(R.m_lpdEntries[1]*R.m_lpdEntries[1] + R.m_lpdEntries[4]*R.m_lpdEntries[4] + R.m_lpdEntries[7]*R.m_lpdEntries[7]);

	r1[0] = R.m_lpdEntries[0]/norm1;
	r1[1] = R.m_lpdEntries[3]/norm1;
	r1[2] = R.m_lpdEntries[6]/norm1;

	r2[0] = R.m_lpdEntries[1]/norm2;
	r2[1] = R.m_lpdEntries[4]/norm2;
	r2[2] = R.m_lpdEntries[7]/norm2;

	// r3 = r1 x r2
	r3[0] = r1[1]*r2[2] - r2[1]*r1[2];
	r3[1] = r1[2]*r2[0] - r1[0]*r2[2];
	r3[2] = r1[0]*r2[1] - r1[1]*r2[0];

	// Make r1 perpendicular to r2
	double theta = acos(r1[0]*r2[0]+r1[1]*r2[1]+r1[2]*r2[2]);
	
	MyMatrix RR(3,3);

	RotationMatrix(-0.5*(0.5*3.141592653589793-theta), r3, RR);

	// update R(:,1)
	R.m_lpdEntries[0] = RR.m_lpdEntries[0]*r1[0] + RR.m_lpdEntries[1]*r1[1] + RR.m_lpdEntries[2]*r1[2];
	R.m_lpdEntries[3] = RR.m_lpdEntries[3]*r1[0] + RR.m_lpdEntries[4]*r1[1] + RR.m_lpdEntries[5]*r1[2];
	R.m_lpdEntries[6] = RR.m_lpdEntries[6]*r1[0] + RR.m_lpdEntries[7]*r1[1] + RR.m_lpdEntries[8]*r1[2];

	RotationMatrix(0.5*(0.5*3.141592653589793-theta), r3, RR);

	// update R(:,2)
	R.m_lpdEntries[1] = RR.m_lpdEntries[0]*r2[0] + RR.m_lpdEntries[1]*r2[1] + RR.m_lpdEntries[2]*r2[2];
	R.m_lpdEntries[4] = RR.m_lpdEntries[3]*r2[0] + RR.m_lpdEntries[4]*r2[1] + RR.m_lpdEntries[5]*r2[2];
	R.m_lpdEntries[7] = RR.m_lpdEntries[6]*r2[0] + RR.m_lpdEntries[7]*r2[1] + RR.m_lpdEntries[8]*r2[2];

	// update R(:,3)
	R.m_lpdEntries[2] = r3[0];
	R.m_lpdEntries[5] = r3[1];
	R.m_lpdEntries[8] = r3[2];
}

/*********************************************************************************************************/
/* Given camera internal parameters, estimate camera pose from homography                                */
/* Because H = K[r1 r2 t] => [r1 r2 t] = inv(K)*H                                                        */
/* Input: H: homography matrix, fx:focal length in x-axis , fy:focal length in y-axis s:skew factor,     */
/*        (ux,uy): principal point                                                                       */
/* Output: t: camera translation, R=[r1 r2 r3]: camera rotation                                          */
/*********************************************************************************************************/
void EstimateCameraPoseFromHomography(MyMatrix &H, double fx, double fy, double s, double ux, double uy, MyMatrix &R, Vector3d &t)
{
	MyMatrix K(3,3);

	K.m_lpdEntries[0] = fx;
	K.m_lpdEntries[1] = s;
	K.m_lpdEntries[2] = ux;
	K.m_lpdEntries[3] = 0.0;
	K.m_lpdEntries[4] = fy;
	K.m_lpdEntries[5] = uy;
	K.m_lpdEntries[6] = 0.0;
	K.m_lpdEntries[7] = 0.0;
	K.m_lpdEntries[8] = 1.0;

	EstimateCameraPoseFromHomography(H,K,R,t);
}

void CostFunctionForCameraRefinement(double *par, double *x, int m, int n, void *data)
{
	double t[3],K[3][3],P[3][4];
	double *pdAddData;

	pdAddData = (double *)(data);
	MyMatrix R(3,3);

	RotationMatrix(par[0], par+1, R);
	t[0] = par[4]; t[1] = par[5]; t[2] = par[6];
	K[0][0] = pdAddData[0];
	K[0][1] = pdAddData[1];
	K[0][2] = pdAddData[2];
	K[1][0] = pdAddData[3];
	K[1][1] = pdAddData[4];
	K[1][2] = pdAddData[5];
	K[2][0] = pdAddData[6];
	K[2][1] = pdAddData[7];
	K[2][2] = pdAddData[8];

	// P = K[R|t]
	P[0][0] = K[0][0]*R.m_lpdEntries[0] + K[0][1]*R.m_lpdEntries[3] + K[0][2]*R.m_lpdEntries[6];
	P[0][1] = K[0][0]*R.m_lpdEntries[1] + K[0][1]*R.m_lpdEntries[4] + K[0][2]*R.m_lpdEntries[7];
	//P[0][2] = K[0][0]*R.m_lpdEntries[2] + K[0][1]*R.m_lpdEntries[5] + K[0][2]*R.m_lpdEntries[8];
	P[0][3] = K[0][0]*t[0] + K[0][1]*t[1] + K[0][2]*t[2];

	P[1][0] = K[1][0]*R.m_lpdEntries[0] + K[1][1]*R.m_lpdEntries[3] + K[1][2]*R.m_lpdEntries[6];
	P[1][1] = K[1][0]*R.m_lpdEntries[1] + K[1][1]*R.m_lpdEntries[4] + K[1][2]*R.m_lpdEntries[7];
	//P[1][2] = K[1][0]*R.m_lpdEntries[2] + K[1][1]*R.m_lpdEntries[5] + K[1][2]*R.m_lpdEntries[8];
	P[1][3] = K[1][0]*t[0] + K[1][1]*t[1] + K[1][2]*t[2];

	P[2][0] = K[2][0]*R.m_lpdEntries[0] + K[2][1]*R.m_lpdEntries[3] + K[2][2]*R.m_lpdEntries[6];
	P[2][1] = K[2][0]*R.m_lpdEntries[1] + K[2][1]*R.m_lpdEntries[4] + K[2][2]*R.m_lpdEntries[7];
	//P[2][2] = K[2][0]*R.m_lpdEntries[2] + K[2][1]*R.m_lpdEntries[5] + K[2][2]*R.m_lpdEntries[8];
	P[2][3] = K[2][0]*t[0] + K[2][1]*t[1] + K[2][2]*t[2];

	double u, v, w;

	// m ~ K[R|t]M = PM
	for(int i=0; i<n/2; i++)
	{
		u = P[0][0] * pdAddData[9+2*i] + P[0][1] * pdAddData[9+2*i+1] + P[0][3];
		v = P[1][0] * pdAddData[9+2*i] + P[1][1] * pdAddData[9+2*i+1] + P[1][3];
		w = P[2][0] * pdAddData[9+2*i] + P[2][1] * pdAddData[9+2*i+1] + P[2][3];
		x[2*i] = u/w;
		x[2*i+1] = v/w;
	}
}

void DebugShowCostForCameraPoseRefinement(Point2d *pPoints1, Point2d *pPoints2, int nMeasurements, double fx, double fy, double s, double ux, double uy, MyMatrix &R, Vector3d &t)
{
	MyMatrix K(3,3);

	K.m_lpdEntries[0] = fx;
	K.m_lpdEntries[1] = s;
	K.m_lpdEntries[2] = ux;
	K.m_lpdEntries[3] = 0.0;
	K.m_lpdEntries[4] = fy;
	K.m_lpdEntries[5] = uy;
	K.m_lpdEntries[6] = 0.0;
	K.m_lpdEntries[7] = 0.0;
	K.m_lpdEntries[8] = 1.0;

	MyMatrix KR(3,3), P(3,4);

	K.Multiplication(&R,&KR);

	P.m_lpdEntries[0] = KR.m_lpdEntries[0];
	P.m_lpdEntries[1] = KR.m_lpdEntries[1];
	//P.m_lpdEntries[2] = KR.m_lpdEntries[2];
	P.m_lpdEntries[3] = K.m_lpdEntries[0]*t.x + K.m_lpdEntries[1]*t.y + K.m_lpdEntries[2]*t.z;

	P.m_lpdEntries[4] = KR.m_lpdEntries[3];
	P.m_lpdEntries[5] = KR.m_lpdEntries[4];
	//P.m_lpdEntries[6] = KR.m_lpdEntries[5];
	P.m_lpdEntries[7] = K.m_lpdEntries[3]*t.x + K.m_lpdEntries[4]*t.y + K.m_lpdEntries[5]*t.z;

	P.m_lpdEntries[8] = KR.m_lpdEntries[6];
	P.m_lpdEntries[9] = KR.m_lpdEntries[7];
	//P.m_lpdEntries[10] = KR.m_lpdEntries[8];
	P.m_lpdEntries[11] = K.m_lpdEntries[6]*t.x + K.m_lpdEntries[7]*t.y + K.m_lpdEntries[8]*t.z;

	double cost=0.0,u,v,w,x,y;

	for(int i=0; i<nMeasurements; i++)
	{
		u = P.m_lpdEntries[0]*pPoints1[i].x + P.m_lpdEntries[1]*pPoints1[i].y + P.m_lpdEntries[3];
		v = P.m_lpdEntries[4]*pPoints1[i].x + P.m_lpdEntries[5]*pPoints1[i].y + P.m_lpdEntries[7];
		w = P.m_lpdEntries[8]*pPoints1[i].x + P.m_lpdEntries[9]*pPoints1[i].y + P.m_lpdEntries[11];

		x = u/w;
		y = v/w;
		cost += (x-pPoints2[i].x)*(x-pPoints2[i].x);
		cost += (y-pPoints2[i].y)*(y-pPoints2[i].y);
	}
	printf("Cost for camera pose refinement = %f\n", cost);
}

/*************************************************************************************************************/
/* Refine camera pose using Levenberg-Marquardt minimization algorithm                                       */
/* Reference: Lepetit and Fua, "Monocular model-based 3D tracking of rigid objects: A survey," Fundations    */
/*           and Trends in Computer Graphics and Vision, Vol.1, No. 1, pp. 1-89, 2005                        */ 
/* Input: pPoints1: 3D points (with z-coordinates zero), pPoints2: measurement image points                  */
/*        nPoints: number of points fx:focal length in x-axis, fy:focal length in y-axis s: skew factor,     */
/*        (ux,uy): principal point, R: initial rotation matrix, t: initial translation vector                */ 
/* Output: R: refined rotation matrix, t:refine translation vector                                           */
/*************************************************************************************************************/
void RefineCameraPose(Point2d *pPoints1, Point2d *pPoints2, int nPoints, double fx, double fy, double s, double ux, double uy, MyMatrix &R, Vector3d &t)
{
	double par[7]; // initial parameters par[0]: rotation angle, par[1]-par[3]:rotation axis par[4]-par[6]: translation vector
	double *pdMeasurements; // Measurements data
	double *pdAddData; // Additional data for cost function
	//double axis[3];

	// Set initial parameters
	GetRotationAxisAndAngle(R,par[0],par+1);
	par[4] = t.x; par[5] = t.y; par[6] = t.z;

	//printf("Initial Rotation Matrix\n");
	//printf("%f %f %f\n",R.m_lpdEntries[0],R.m_lpdEntries[1],R.m_lpdEntries[2]);
	//printf("%f %f %f\n",R.m_lpdEntries[3],R.m_lpdEntries[4],R.m_lpdEntries[5]);
	//printf("%f %f %f\n",R.m_lpdEntries[6],R.m_lpdEntries[7],R.m_lpdEntries[8]);
	//printf("Initial Translation\n");
	//printf("%f %f %f\n",t.x,t.y,t.z);

	//DebugShowCostForCameraPoseRefinement(pPoints1, pPoints2, nPoints, fx, fy, s, ux, uy, R, t);

	pdMeasurements = new double[2*nPoints];
	pdAddData = new double[2*nPoints+9]; // 2n for pPoints1, 9 for matrix K

	pdAddData[0] = fx;
	pdAddData[1] = s;
	pdAddData[2] = ux;
	pdAddData[3] = 0.0;
	pdAddData[4] = fy;
	pdAddData[5] = uy;
	pdAddData[6] = 0.0;
	pdAddData[7] = 0.0;
	pdAddData[8] = 1.0;

	for(int i=0; i<nPoints; i++)
	{
		pdAddData[2*i+9] = pPoints1[i].x;
		pdAddData[2*i+1+9] = pPoints1[i].y;
		pdMeasurements[2*i] = pPoints2[i].x;
		pdMeasurements[2*i+1] = pPoints2[i].y;
	}

	double info[LM_INFO_SZ];
	int ret=dlevmar_dif(CostFunctionForCameraRefinement, par, pdMeasurements, 7, 2*nPoints, 1000, NULL, info, NULL, NULL, pdAddData);  // no Jacobian
	//printf("Initial Cost=%f Final Cost=%f\n",info[0],info[1]);
	//printf("# Iter= %d\n",ret);

	delete [] pdMeasurements;
	delete [] pdAddData;
	
	RotationMatrix(par[0], par+1, R);
	t.x = par[4]; t.y = par[5]; t.z = par[6];

	//DebugShowCostForCameraPoseRefinement(pPoints1, pPoints2, nPoints, fx, fy, s, ux, uy, R, t);

	//printf("Final Rotation Matrix\n");
	//printf("%f %f %f\n",R.m_lpdEntries[0],R.m_lpdEntries[1],R.m_lpdEntries[2]);
	//printf("%f %f %f\n",R.m_lpdEntries[3],R.m_lpdEntries[4],R.m_lpdEntries[5]);
	//printf("%f %f %f\n",R.m_lpdEntries[6],R.m_lpdEntries[7],R.m_lpdEntries[8]);
	//printf("Final Translation\n");
	//printf("%f %f %f\n",t.x,t.y,t.z);
}

void RefineCameraPose(std::vector<cv::Point2f> pPoints1, std::vector<cv::Point2f> pPoints2, int nPoints, double fx, double fy, double s, double ux, double uy, MyMatrix &R, Vector3d &t)
{
	double par[7]; // initial parameters par[0]: rotation angle, par[1]-par[3]:rotation axis par[4]-par[6]: translation vector
	double *pdMeasurements; // Measurements data
	double *pdAddData; // Additional data for cost function
	//double axis[3];

	// Set initial parameters
	GetRotationAxisAndAngle(R, par[0], par + 1);
	par[4] = t.x; par[5] = t.y; par[6] = t.z;

	//printf("Initial Rotation Matrix\n");
	//printf("%f %f %f\n",R.m_lpdEntries[0],R.m_lpdEntries[1],R.m_lpdEntries[2]);
	//printf("%f %f %f\n",R.m_lpdEntries[3],R.m_lpdEntries[4],R.m_lpdEntries[5]);
	//printf("%f %f %f\n",R.m_lpdEntries[6],R.m_lpdEntries[7],R.m_lpdEntries[8]);
	//printf("Initial Translation\n");
	//printf("%f %f %f\n",t.x,t.y,t.z);

	//DebugShowCostForCameraPoseRefinement(pPoints1, pPoints2, nPoints, fx, fy, s, ux, uy, R, t);

	pdMeasurements = new double[2 * nPoints];
	pdAddData = new double[2 * nPoints + 9]; // 2n for pPoints1, 9 for matrix K

	pdAddData[0] = fx;
	pdAddData[1] = s;
	pdAddData[2] = ux;
	pdAddData[3] = 0.0;
	pdAddData[4] = fy;
	pdAddData[5] = uy;
	pdAddData[6] = 0.0;
	pdAddData[7] = 0.0;
	pdAddData[8] = 1.0;

	for (int i = 0; i<nPoints; i++)
	{
		pdAddData[2 * i + 9] = pPoints1[i].x;
		pdAddData[2 * i + 1 + 9] = pPoints1[i].y;
		pdMeasurements[2 * i] = pPoints2[i].x;
		pdMeasurements[2 * i + 1] = pPoints2[i].y;
	}

	double info[LM_INFO_SZ];
	int ret = dlevmar_dif(CostFunctionForCameraRefinement, par, pdMeasurements, 7, 2 * nPoints, 1000, NULL, info, NULL, NULL, pdAddData);  // no Jacobian
	//printf("Initial Cost=%f Final Cost=%f\n",info[0],info[1]);
	//printf("# Iter= %d\n",ret);

	delete[] pdMeasurements;
	delete[] pdAddData;

	RotationMatrix(par[0], par + 1, R);
	t.x = par[4]; t.y = par[5]; t.z = par[6];

	//DebugShowCostForCameraPoseRefinement(pPoints1, pPoints2, nPoints, fx, fy, s, ux, uy, R, t);

	//printf("Final Rotation Matrix\n");
	//printf("%f %f %f\n",R.m_lpdEntries[0],R.m_lpdEntries[1],R.m_lpdEntries[2]);
	//printf("%f %f %f\n",R.m_lpdEntries[3],R.m_lpdEntries[4],R.m_lpdEntries[5]);
	//printf("%f %f %f\n",R.m_lpdEntries[6],R.m_lpdEntries[7],R.m_lpdEntries[8]);
	//printf("Final Translation\n");
	//printf("%f %f %f\n",t.x,t.y,t.z);
}

/***********************************************************************************************************************/
/* Update camera focal length using Homography                                                                         */
/* Reference: G. Simon, A. W. Fitzgibbon, and A. Zisserman, "Markerless tracking using planar structures               */ 
/*            in the scene," IEEE and ACM International Symposium on Augmented Reality, pp. 120 - 128, 2000.           */ 
/*            & 2011年國科會計畫書                                                                                     */  
/* Input: cameraPara: camera intrinsic parameters                                                                      */
/*        H: homography between the planes                                                                             */   
/* Output: cameraPara: camera intrinsic paramters, especially camera focal length (camerPara[0] & cameraPara[4])       */
/***********************************************************************************************************************/
bool UpdateCameraIntrinsicParameters(double *cameraPara, MyMatrix &H)
{
	double v[3], w[3];

	// v= H*e1 = H*[1 0 0]'
	v[0] = H.m_lpdEntries[0];
	v[1] = H.m_lpdEntries[3];
	v[2] = H.m_lpdEntries[6];

	// w= H*e2 = H*[0 1 0]'
	w[0] = H.m_lpdEntries[1];
	w[1] = H.m_lpdEntries[4];
	w[2] = H.m_lpdEntries[7];

	// re-scale
	v[0]/=v[2]; v[1]/=v[2];
	w[0]/=w[2]; w[1]/=w[2];

	// translate image principal point to (0, 0)
	v[0]-=cameraPara[2];
	v[1]-=cameraPara[5];

	w[0]-=cameraPara[2];
	w[1]-=cameraPara[5];

	double inner = -v[0]*w[0] - v[1]*w[1];

	if(inner < 0)
	{
		printf("Fail to estimate Camera Focal Length\n");
		return false;
	}
	else
	{
		cameraPara[0] = cameraPara[4] = sqrt(inner);
		printf("Camera Focal Length = %f\n",sqrt(inner));
		return true;
	}
}

void CreateFeatureMap(FeatureMap &featureMap, int minHessian)
{
	SurfDetection(featureMap.image, featureMap.keypoints, featureMap.descriptors, minHessian);

	for (std::size_t i = 0; i < featureMap.keypoints.size(); ++i)
	{
		featureMap.keypoints[i].pt.x -= featureMap.image.cols / 2;
		featureMap.keypoints[i].pt.y -= featureMap.image.rows / 2;
		featureMap.keypoints[i].pt.y = -featureMap.keypoints[i].pt.y; // Because image y coordinate is positive in downward direction
	}
}

/*void Debug3D(KeyFrame &image, char *name, double *cameraPara)
{
	cv::Mat K(3, 3, CV_64F);
	for (int i = 0; i < 9; ++i)
		K.at<double>(i) = cameraPara[i];
	cv::Mat Point4D(4, 1, CV_64F);
	cv::Mat uvw(3, 1, CV_64F);
	cv::Mat mat(3, 4, CV_64F);
	Point4D.at<double>(0) = 400.0;
	Point4D.at<double>(1) = 300.0;
	Point4D.at<double>(2) = 0.0;
	Point4D.at<double>(3) = 1.0;
	//cout << K << endl << image.projMatrix << endl;
	//mat = K * image.projMatrix;
	//uvw = K * image.projMatrix * Point4D;
	float x, y;
	x = (float)uvw.at<double>(0) / uvw.at<double>(2);
	y = (float)uvw.at<double>(1) / uvw.at<double>(2);
	cout << "Debug 3D :" << x << ", " << y << endl;
	std::vector<cv::KeyPoint> points;
	cv::KeyPoint point;
	point.pt.x = x;
	point.pt.y = y;
	points.push_back(point);
	cv::Mat out;
	cv::drawKeypoints(image.image, points, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	imshow(name, out);
	imwrite(name, out);
	waitKey(0);
}*/

bool EstimateCameraTransformation(unsigned long FrameCount, double *cameraPara, double trans[3][4], FeatureMap &featureMap, Frame &currFrame, cv::Mat &currFrameImg, std::vector<KeyFrame> &keyFrames, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &currFrameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapInliers, std::vector<cv::Point2f> &prevFrameInliers)
{
	//Using homography to estimate camera pose

	cv::Mat mask;	//Record the inliers 
	cv::Mat Homo(cv::findHomography(featureMapGoodMatches, currFrameGoodMatches, CV_RANSAC, 3.0, mask));
	MyMatrix H(3, 3);
	for (int i = 0; i < 9; ++i)
		H.m_lpdEntries[i] = Homo.at<double>(i);
	
	std::vector<cv::Point2f> featureMapInliers, frameInliers;
	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i))
		{
			featureMapInliers.push_back(featureMapGoodMatches[i]);
			frameInliers.push_back(currFrameGoodMatches[i]);
		}
	}

	MyMatrix R(3, 3);
	Vector3d t;
	EstimateCameraPoseFromHomography(H, cameraPara[0], cameraPara[4], cameraPara[1], cameraPara[2], cameraPara[5], R, t);
	RefineCameraPose(featureMapInliers, frameInliers, frameInliers.size(), cameraPara[0], cameraPara[4], cameraPara[1], cameraPara[2], cameraPara[5], R, t);

	trans[0][0] = R.m_lpdEntries[0];
	trans[0][1] = R.m_lpdEntries[1];
	trans[0][2] = R.m_lpdEntries[2];
	trans[0][3] = t.x;
	trans[1][0] = R.m_lpdEntries[3];
	trans[1][1] = R.m_lpdEntries[4];
	trans[1][2] = R.m_lpdEntries[5];
	trans[1][3] = t.y;
	trans[2][0] = R.m_lpdEntries[6];
	trans[2][1] = R.m_lpdEntries[7];
	trans[2][2] = R.m_lpdEntries[8];
	trans[2][3] = t.z;

	//	把影像中好的特徵點放入prevFeatureMapinliers ，給OpticalFlow計算用
	//	把場景中好的特徵點放入prevFrameinliers，給OpticalFlow計算用
	prevFeatureMapInliers.swap(featureMapInliers);
	prevFrameInliers.swap(frameInliers);

	std::vector<cv::Point2f> ().swap(featureMapGoodMatches);
	std::vector<cv::Point2f> ().swap(currFrameGoodMatches);
	std::vector<cv::Point2f>().swap(featureMapInliers);
	std::vector<cv::Point2f>().swap(frameInliers);
	return true;
}

bool EstimateCameraTransformation(double *cameraPara, std::vector<KeyFrame> &keyFrames, Frame &currFrame, cv::Mat &currFrameImg, std::vector<cv::DMatch> &goodMatchesSet)
{
	//3D map
	//cv::solvePnPRansac();
}
