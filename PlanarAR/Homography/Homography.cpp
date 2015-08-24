#include <iostream>
#include <cmath>
#include "..\BasicType.h"
#include "..\Matrix\MyMatrix.h"
#include "..\MathLib\MathLib.h"

void RotationMatrix(double angle, double axis[3], MyMatrix &R);
void RefineHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, BYTE *pbInliers, MyMatrix &H);

/*******************************************************************************************************/
/* Random select M integer numbers from [1,N]                                                          */
/*******************************************************************************************************/
void RandomSelect(int N, int M, int *piSelect)
{
	int i, j, d, coll;

	if (N>RAND_MAX)
	{
		printf("RandomSelect error!!!\n");
		return;
	}

	piSelect[0] = rand() % N;

	i = 1;
	while (i < M)
	{
		coll = 0;
		d = rand() % N;
		for (j = 0; j < i; j++)
		{
			if (d == piSelect[j])
			{
				coll = 1;
				break;
			}
		}
		if (coll == 0)
		{
			piSelect[i++] = d;
		}
	}
}


/*******************************************************************************************************/
/* Compute homography using direct linear transformation                                               */
/* x2=Hx1; x2: pPoints2, x1: pPoints1, H: homography matrix                                            */
/* (See "Multiple View Geometry in Computer Vision", page 88-91)                                       */ 
/*******************************************************************************************************/
bool Homography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H)
{
	if (nPoints < 4) return false;
	MyMatrix A(2*nPoints,9);

	for(int i=0; i<nPoints; ++i)
	{
		// first row
		A.m_lpdEntries[18*i+0] = 0.0;
		A.m_lpdEntries[18*i+1] = 0.0;
		A.m_lpdEntries[18*i+2] = 0.0;
		A.m_lpdEntries[18*i+3] = -pPoints1[i].x;
		A.m_lpdEntries[18*i+4] = -pPoints1[i].y;
		A.m_lpdEntries[18*i+5] = -1.0;
		A.m_lpdEntries[18*i+6] = pPoints2[i].y*pPoints1[i].x;
		A.m_lpdEntries[18*i+7] = pPoints2[i].y*pPoints1[i].y;
		A.m_lpdEntries[18*i+8] = pPoints2[i].y;

		// second row
		A.m_lpdEntries[18*i+9] = pPoints1[i].x;
		A.m_lpdEntries[18*i+10] = pPoints1[i].y;
		A.m_lpdEntries[18*i+11] = 1.0;
		A.m_lpdEntries[18*i+12] = 0.0;
		A.m_lpdEntries[18*i+13] = 0.0;
		A.m_lpdEntries[18*i+14] = 0.0;
		A.m_lpdEntries[18*i+15] = -pPoints2[i].x*pPoints1[i].x;
		A.m_lpdEntries[18*i+16] = -pPoints2[i].x*pPoints1[i].y;
		A.m_lpdEntries[18*i+17] = -pPoints2[i].x;
	}

	MyMatrix U(2*nPoints,2*nPoints),S(2*nPoints,9),V(9,9);

	A.SVD(&U,&S,&V);

	for(int i=0; i<9; ++i)
	{
		H.m_lpdEntries[i] = V.m_lpdEntries[9*i+8];
	}
	return true;
}

/*******************************************************************************************************/
/* Compute homography with normalization                                                               */
/* x2=Hx1; x2: pPoints2, x1: pPoints1, H: homography matrix                                            */
/* (See "Multiple View Geometry in Computer Vision", page 107-110)                                     */ 
/*******************************************************************************************************/
bool HomographyWithNormalization(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H)
{
    Point2d *pM1,*pM2,Mean1,Mean2;
    double AvgDis1,AvgDis2,Scale1,Scale2;

	if (nPoints < 4) return false;

    // Do normalization
    Mean1.x=Mean1.y=Mean2.x=Mean2.y=0.0;
    for(int i=0;i<nPoints;++i)
    {
        Mean1.x+=pPoints1[i].x;
        Mean1.y+=pPoints1[i].y;
        Mean2.x+=pPoints2[i].x;
        Mean2.y+=pPoints2[i].y;
    }
    Mean1.x/=nPoints;
    Mean1.y/=nPoints;
    Mean2.x/=nPoints;
    Mean2.y/=nPoints;

    pM1 = new Point2d[nPoints];
    pM2 = new Point2d[nPoints];
    AvgDis1=AvgDis2=0.0;
    for(int i=0;i<nPoints;++i)
    {
        pM1[i].x=pPoints1[i].x-Mean1.x;
        pM1[i].y=pPoints1[i].y-Mean1.y;
        pM2[i].x=pPoints2[i].x-Mean2.x;
        pM2[i].y=pPoints2[i].y-Mean2.y;
        AvgDis1+=sqrt(pM1[i].x*pM1[i].x+pM1[i].y*pM1[i].y);
        AvgDis2+=sqrt(pM2[i].x*pM2[i].x+pM2[i].y*pM2[i].y);
    }
    AvgDis1/=nPoints;
    AvgDis2/=nPoints;
    Scale1=sqrt(2.0)/AvgDis1;
    Scale2=sqrt(2.0)/AvgDis2;
    for(int i=0;i<nPoints;++i)
    {
        pM1[i].x*=Scale1;
        pM1[i].y*=Scale1;
        pM2[i].x*=Scale2;
        pM2[i].y*=Scale2;
    }

	MyMatrix NH(3,3);

	if (Homography(pM1, pM2, nPoints, NH) == false) return false;

    // Assign to H and modify it according to the normalization
    // H = inv(T2)*NH*T1
    // T1=[Scale1 0 -Scale1*Mean1.x; 0 Scale1 -Scale1*Mean1.y; 0 0 1]
    // T2=[Scale2 0 -Scale2*Mean2.x; 0 Scale2 -Scale2*Mean2.y; 0 0 1]
	// invT2=[1/Scale2 0 Mean2.x; 0 1/Scale2 Mean2.y; 0 0 1]

	MyMatrix T1(3,3),invT2(3,3);
	T1.m_lpdEntries[0] = Scale1; T1.m_lpdEntries[1] = 0.0; T1.m_lpdEntries[2] = -Scale1*Mean1.x;
	T1.m_lpdEntries[3] = 0.0; T1.m_lpdEntries[4] = Scale1; T1.m_lpdEntries[5] = -Scale1*Mean1.y;
	T1.m_lpdEntries[6] = 0.0; T1.m_lpdEntries[7] = 0.0; T1.m_lpdEntries[8] = 1.0;

	invT2.m_lpdEntries[0] = 1.0/Scale2; invT2.m_lpdEntries[1] = 0.0; invT2.m_lpdEntries[2] = Mean2.x;
	invT2.m_lpdEntries[3] = 0.0; invT2.m_lpdEntries[4] = 1.0/Scale2; invT2.m_lpdEntries[5] = Mean2.y;
	invT2.m_lpdEntries[6] = 0.0; invT2.m_lpdEntries[7] = 0.0; invT2.m_lpdEntries[8] = 1.0;

	MyMatrix NHT1(3,3);
	NH.Multiplication(&T1,&NHT1); // NHT1=NH*T1
	invT2.Multiplication(&NHT1,&H);

    delete [] pM1;
    delete [] pM2;

	return true;
}

/*******************************************************************************************************/
/* Robust computation of homography matrix (using Least Median Squares)                                */
/* x2=Hx1; x2: pPoints2, x1: pPoints1, H: homography matrix                                            */
/* return value: number of inliers                                                                     */
/*******************************************************************************************************/
int LMedSHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H, BYTE *pbInliers)
{
    if(nPoints < 4) return -1;

    int J=0, piSelect[4];
	double *pdDis,MinMed,Median;
	Point2d pP1[4],pP2[4];
	MyMatrix HH(3,3);

	//srand(0);

	int SN = (int)(log(0.01)/log(1-pow(1-0.5,4)))+1; // number of trials

	pdDis = new double[nPoints];

	MinMed=1000000000000000;
    while(J<SN)
    {
		RandomSelect(nPoints,4,piSelect);

		pP1[0].x=pPoints1[piSelect[0]].x; pP1[0].y=pPoints1[piSelect[0]].y;
		pP1[1].x=pPoints1[piSelect[1]].x; pP1[1].y=pPoints1[piSelect[1]].y;
		pP1[2].x=pPoints1[piSelect[2]].x; pP1[2].y=pPoints1[piSelect[2]].y;
		pP1[3].x=pPoints1[piSelect[3]].x; pP1[3].y=pPoints1[piSelect[2]].y;
		pP2[0].x=pPoints2[piSelect[0]].x; pP2[0].y=pPoints2[piSelect[0]].y;
		pP2[1].x=pPoints2[piSelect[1]].x; pP2[1].y=pPoints2[piSelect[1]].y;
		pP2[2].x=pPoints2[piSelect[2]].x; pP2[2].y=pPoints2[piSelect[2]].y;
		pP2[3].x=pPoints2[piSelect[3]].x; pP2[3].y=pPoints2[piSelect[2]].y;

		if (HomographyWithNormalization(pP1, pP2, 4, HH) == false)
		// if (Homography(pP1, pP2, 4, HH) == false);
		{
			delete[] pdDis;
			return 0;
		}

		for(int i=0;i<nPoints;++i)
		{
			double u = pPoints1[i].x * HH.m_lpdEntries[0] + pPoints1[i].y * HH.m_lpdEntries[1] + HH.m_lpdEntries[2];
			double v = pPoints1[i].x * HH.m_lpdEntries[3] + pPoints1[i].y * HH.m_lpdEntries[4] + HH.m_lpdEntries[5];
			double w = pPoints1[i].x * HH.m_lpdEntries[6] + pPoints1[i].y * HH.m_lpdEntries[7] + HH.m_lpdEntries[8];
			double x = u/w;
			double y = v/w;
			pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);
		}

		Median=Mediand(pdDis,nPoints);

        if(Median<MinMed)
        {
            MinMed=Median;
            for(int j=0;j<9;++j) H.m_lpdEntries[j]=HH.m_lpdEntries[j];
        }
		J++;
	}

    MyMatrix CH(3,3);
    for(int i=0;i<9;++i) CH.m_lpdEntries[i]=H.m_lpdEntries[i];

	Point2d *pSelectPt1,*pSelectPt2;

    pSelectPt1 = new Point2d[nPoints];
    pSelectPt2 = new Point2d[nPoints];
    Median=MinMed;

	J=0;
    while(J<100)
    {
        int N=0;
        double Sigma2=1.4826*(1.0+5.0/(nPoints-4))*sqrt(Median);
        Sigma2*=Sigma2;

        for(int i=0;i<nPoints;++i)
        {
			double u = pPoints1[i].x * H.m_lpdEntries[0] + pPoints1[i].y * H.m_lpdEntries[1] + H.m_lpdEntries[2];
			double v = pPoints1[i].x * H.m_lpdEntries[3] + pPoints1[i].y * H.m_lpdEntries[4] + H.m_lpdEntries[5];
			double w = pPoints1[i].x * H.m_lpdEntries[6] + pPoints1[i].y * H.m_lpdEntries[7] + H.m_lpdEntries[8];
			double x = u/w;
			double y = v/w;

			pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);

            if(pdDis[i]<=6.25*Sigma2)
            {
                pbInliers[i]=1;
				pSelectPt1[N].x = pPoints1[i].x;
				pSelectPt1[N].y = pPoints1[i].y;
				pSelectPt2[N].x = pPoints2[i].x;
				pSelectPt2[N].y = pPoints2[i].y;
                N++;
            }
            else
                pbInliers[i]=0;
        }
		if (HomographyWithNormalization(pSelectPt1, pSelectPt2, N, H) == false)
		{
			delete [] pdDis;
			delete [] pSelectPt1;
			delete [] pSelectPt2;
			return 0;
		}
        double tmp=0.0;
        for(int i=0;i<9;++i) tmp+=(H.m_lpdEntries[i]-CH.m_lpdEntries[i])*(H.m_lpdEntries[i]-CH.m_lpdEntries[i]);

        if(tmp>0.01)
        {
            for(int i=0;i<9;++i) CH.m_lpdEntries[i]=H.m_lpdEntries[i];
            Median=Mediand(pdDis,nPoints);
        }
        else
            break;
		J++;
    }

    delete [] pSelectPt1;
    delete [] pSelectPt2;

    for(int i=0;i<nPoints;++i)
    {
		double u = pPoints1[i].x * H.m_lpdEntries[0] + pPoints1[i].y * H.m_lpdEntries[1] + H.m_lpdEntries[2];
		double v = pPoints1[i].x * H.m_lpdEntries[3] + pPoints1[i].y * H.m_lpdEntries[4] + H.m_lpdEntries[5];
		double w = pPoints1[i].x * H.m_lpdEntries[6] + pPoints1[i].y * H.m_lpdEntries[7] + H.m_lpdEntries[8];
		double x = u/w;
		double y = v/w;

		pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);
    }
    Median=Mediand(pdDis,nPoints);

    double Sigma2=1.4826*(1.0+5.0/(nPoints-4))*sqrt(Median);
    Sigma2*=Sigma2;
	int N=0;
    for(int i=0;i<nPoints;++i)
    {
        if(pdDis[i]<=6.25*Sigma2)
		{
            pbInliers[i]=1;
			N++;
		}
        else
            pbInliers[i]=0;
    }

	RefineHomography(pPoints1,pPoints2,nPoints,pbInliers,H);

    delete [] pdDis;
	return N;
}

/*******************************************************************************************************/
/* Robust computation of homography matrix (using adaptive Least Median Squares)                       */
/* x2=Hx1; x2: pPoints2, x1: pPoints1, H: homography matrix                                            */
/* return value: number of inliers                                                                     */
/*******************************************************************************************************/
int AdaptiveLMedSHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H, BYTE *pbInliers)
{
    int J=0, piSelect[4];
	double *pdDis,MinMed,Median;
	Point2d pP1[4],pP2[4];
	MyMatrix HH(3,3);

	//srand(0);

	int SN = (int)(log(0.01)/log(1-pow(1-0.5,4)))+1; // number of trials

	pdDis = new double[nPoints];

	MinMed=1000000000000000;
    while(J<SN)
    {
		RandomSelect(nPoints,4,piSelect);

		pP1[0].x=pPoints1[piSelect[0]].x; pP1[0].y=pPoints1[piSelect[0]].y;
		pP1[1].x=pPoints1[piSelect[1]].x; pP1[1].y=pPoints1[piSelect[1]].y;
		pP1[2].x=pPoints1[piSelect[2]].x; pP1[2].y=pPoints1[piSelect[2]].y;
		pP1[3].x=pPoints1[piSelect[3]].x; pP1[3].y=pPoints1[piSelect[2]].y;
		pP2[0].x=pPoints2[piSelect[0]].x; pP2[0].y=pPoints2[piSelect[0]].y;
		pP2[1].x=pPoints2[piSelect[1]].x; pP2[1].y=pPoints2[piSelect[1]].y;
		pP2[2].x=pPoints2[piSelect[2]].x; pP2[2].y=pPoints2[piSelect[2]].y;
		pP2[3].x=pPoints2[piSelect[3]].x; pP2[3].y=pPoints2[piSelect[2]].y;

		if(HomographyWithNormalization(pP1,pP2,4,HH)==false) return 0;
		//if(Homography(pP1,pP2,4,HH)==false) return0;

		for(int i=0;i<nPoints;++i)
		{
			double u = pPoints1[i].x * HH.m_lpdEntries[0] + pPoints1[i].y * HH.m_lpdEntries[1] + HH.m_lpdEntries[2];
			double v = pPoints1[i].x * HH.m_lpdEntries[3] + pPoints1[i].y * HH.m_lpdEntries[4] + HH.m_lpdEntries[5];
			double w = pPoints1[i].x * HH.m_lpdEntries[6] + pPoints1[i].y * HH.m_lpdEntries[7] + HH.m_lpdEntries[8];
			double x = u/w;
			double y = v/w;
			pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);
		}

		Median=Mediand(pdDis,nPoints);

		double Sigma2=1.4826*(1.0+5.0/(nPoints-4))*sqrt(Median);
		Sigma2*=Sigma2;

		int nInliers = 0;
		for(int j=0;j<nPoints;++j)
		{
			if(pdDis[j]<=6.25*Sigma2)
			{
				nInliers++;
			}
		}
		double proportion = 1.0-(double)nInliers/(double)nPoints;
		int new_SN = (int)(log(0.01)/log(1-pow(1-proportion,4)))+1; // number of trials
		if(new_SN<SN) SN = new_SN;

        if(Median<MinMed)
        {
            MinMed=Median;
            for(int j=0;j<9;++j) H.m_lpdEntries[j]=HH.m_lpdEntries[j];
        }
		J++;
	}

    MyMatrix CH(3,3);
    for(int i=0;i<9;++i) CH.m_lpdEntries[i]=H.m_lpdEntries[i];

	Point2d *pSelectPt1,*pSelectPt2;

    pSelectPt1 = new Point2d[nPoints];
    pSelectPt2 = new Point2d[nPoints];
    Median=MinMed;

	J=0;
    while(J<100)
    {
        int N=0;
        double Sigma2=1.4826*(1.0+5.0/(nPoints-4))*sqrt(Median);
        Sigma2*=Sigma2;

        for(int i=0;i<nPoints;++i)
        {
			double u = pPoints1[i].x * H.m_lpdEntries[0] + pPoints1[i].y * H.m_lpdEntries[1] + H.m_lpdEntries[2];
			double v = pPoints1[i].x * H.m_lpdEntries[3] + pPoints1[i].y * H.m_lpdEntries[4] + H.m_lpdEntries[5];
			double w = pPoints1[i].x * H.m_lpdEntries[6] + pPoints1[i].y * H.m_lpdEntries[7] + H.m_lpdEntries[8];
			double x = u/w;
			double y = v/w;

			pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);

            if(pdDis[i]<=6.25*Sigma2)
            {
                pbInliers[i]=1;
				pSelectPt1[N].x = pPoints1[i].x;
				pSelectPt1[N].y = pPoints1[i].y;
				pSelectPt2[N].x = pPoints2[i].x;
				pSelectPt2[N].y = pPoints2[i].y;
                N++;
            }
            else
                pbInliers[i]=0;
        }
		if (HomographyWithNormalization(pSelectPt1, pSelectPt2, N, H) == false)
		{
			delete [] pdDis;
			delete [] pSelectPt1;
			delete [] pSelectPt2;
			return 0;
		}
        double tmp=0.0;
        for(int i=0;i<9;++i) tmp+=(H.m_lpdEntries[i]-CH.m_lpdEntries[i])*(H.m_lpdEntries[i]-CH.m_lpdEntries[i]);

        if(tmp>0.01)
        {
            for(int i=0;i<9;++i) CH.m_lpdEntries[i]=H.m_lpdEntries[i];
            Median=Mediand(pdDis,nPoints);
        }
        else
            break;
		J++;
    }

    delete [] pSelectPt1;
    delete [] pSelectPt2;

    for(int i=0;i<nPoints;++i)
    {
		double u = pPoints1[i].x * H.m_lpdEntries[0] + pPoints1[i].y * H.m_lpdEntries[1] + H.m_lpdEntries[2];
		double v = pPoints1[i].x * H.m_lpdEntries[3] + pPoints1[i].y * H.m_lpdEntries[4] + H.m_lpdEntries[5];
		double w = pPoints1[i].x * H.m_lpdEntries[6] + pPoints1[i].y * H.m_lpdEntries[7] + H.m_lpdEntries[8];
		double x = u/w;
		double y = v/w;

		pdDis[i] = (x-pPoints2[i].x)*(x-pPoints2[i].x)+(y-pPoints2[i].y)*(y-pPoints2[i].y);
    }
    Median=Mediand(pdDis,nPoints);

    double Sigma2=1.4826*(1.0+5.0/(nPoints-4))*sqrt(Median);
    Sigma2*=Sigma2;
	int N=0;
    for(int i=0;i<nPoints;++i)
    {
        if(pdDis[i]<=6.25*Sigma2)
		{
            pbInliers[i]=1;
			N++;
		}
        else
            pbInliers[i]=0;
    }
	RefineHomography(pPoints1,pPoints2,nPoints,pbInliers,H);

    delete [] pdDis;
	return N;
}

#include "..\levmar.h"

void DebugShowCostForHomographyRefinement(double *par, double *measurements, int m, int n, void *data)
{
	double *pdAddData;

	pdAddData = (double *)(data);

	double u, v, w, cost=0.0,x,y;

	for (int i = 0; i<n; i++)
	{
		u = par[0] * pdAddData[2 * i] + par[1] * pdAddData[2 * i + 1] + par[2];
		v = par[3] * pdAddData[2 * i] + par[4] * pdAddData[2 * i + 1] + par[5];
		w = par[6] * pdAddData[2 * i] + par[7] * pdAddData[2 * i + 1] + par[8];
		x = u / w;
		y = v / w;
		cost += (x - measurements[2 * i])*(x - measurements[2 * i]);
		cost += (y - measurements[2 * i+1])*(y - measurements[2 * i+1]);
	}
	printf("Cost for Homography Refinement = %g\n", cost);
}

void VectorGeneratedByHomography(double *par, double *x, int m, int n, void *data)
{
	double *pdAddData;

	pdAddData = (double *)(data);

	double u,v,w;

	for(int i=0; i<n/2; i++)
	{
		u = par[0] * pdAddData[2*i] + par[1] * pdAddData[2*i+1] + par[2];
		v = par[3] * pdAddData[2*i] + par[4] * pdAddData[2*i+1] + par[5];
		w = par[6] * pdAddData[2*i] + par[7] * pdAddData[2*i+1] + par[8];
		x[2*i] = u/w;
		x[2*i+1] = v/w;
	}
}

void JacobianForHomographyRefinement(double *par, double *jacobian, int m, int n, void *data)
{
	double *pdAddData;

	pdAddData = (double *)(data);

	for (int i = 0; i<n / 2; i++)
	{
		double x = pdAddData[2 * i], y = pdAddData[2 * i + 1];
		double denom = par[6] * x + par[7] * y + par[8];

		jacobian[18 * i] = x / denom;
		jacobian[18 * i + 1] = y / denom;
		jacobian[18 * i + 2] = 1.0 / denom;
		jacobian[18 * i + 3] = 0.0;
		jacobian[18 * i + 4] = 0.0;
		jacobian[18 * i + 5] = 0.0;
		jacobian[18 * i + 6] = -(par[0] * x + par[1] * y + par[2])*x / (denom*denom);
		jacobian[18 * i + 7] = -(par[0] * x + par[1] * y + par[2])*y / (denom*denom);
		jacobian[18 * i + 8] = -(par[0] * x + par[1] * y + par[2]) / (denom*denom);

		jacobian[18 * i + 9] = 0.0;
		jacobian[18 * i + 10] = 0.0;
		jacobian[18 * i + 11] = 0.0;
		jacobian[18 * i + 12] = x / denom;
		jacobian[18 * i + 13] = y / denom;
		jacobian[18 * i + 14] = 1.0;
		jacobian[18 * i + 15] = -(par[3] * x + par[4] * y + par[5])*x / (denom*denom);
		jacobian[18 * i + 16] = -(par[3] * x + par[4] * y + par[5])*y / (denom*denom);
		jacobian[18 * i + 17] = -(par[3] * x + par[4] * y + par[5]) / (denom*denom);
	}
}

/*************************************************************************************************************/
/* Refine homography using Levenberg-Marquardt minimization algorithm                                        */
/* Reference: Multiple View Geometry in Computer Vision                                                      */ 
/* Input: pPoints1: points in one plane, pPoints2: mpoints in another plane                                  */
/*        nPoints: number of points, pbInliers: inliers of the corresponding poins                           */
/*        H: initial homography                                                                              */ 
/* Output: H: refined homography                                                                             */
/*************************************************************************************************************/
void RefineHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, BYTE *pbInliers, MyMatrix &H)
{
	int nInliers=0;

	for(int i=0; i<nPoints; ++i)
	{
		if(pbInliers[i]==1) nInliers++;
	}

	double par[9]; // initial parameters 
	double *pdMeasurements; // Measurements data
	double *pdAddData; // Additional data for cost function

	par[0] = H.m_lpdEntries[0];
	par[1] = H.m_lpdEntries[1];
	par[2] = H.m_lpdEntries[2];
	par[3] = H.m_lpdEntries[3];
	par[4] = H.m_lpdEntries[4];
	par[5] = H.m_lpdEntries[5];
	par[6] = H.m_lpdEntries[6];
	par[7] = H.m_lpdEntries[7];
	par[8] = H.m_lpdEntries[8];

	pdMeasurements = new double[2*nInliers]; // pPoints2
	pdAddData = new double[2*nInliers]; // pPoints1

	for(int i=0, j=0; i<nPoints; i++)
	{
		if(pbInliers[i]==1)
		{
			pdAddData[2*j] = pPoints1[i].x;
			pdAddData[2*j+1] = pPoints1[i].y;
			pdMeasurements[2*j] = pPoints2[i].x;
			pdMeasurements[2*j+1] = pPoints2[i].y;
			j++;
		}
	}

	//DebugShowCostForHomographyRefinement(par, pdMeasurements, 9, nInliers, pdAddData);

	double info[LM_INFO_SZ];
	//int ret=dlevmar_dif(VectorGeneratedByHomography, par, pdMeasurements, 9, 2*nInliers, 1000, NULL, info, NULL, NULL, pdAddData);  // no Jacobian
	int ret = dlevmar_der(VectorGeneratedByHomography, JacobianForHomographyRefinement, par, pdMeasurements, 9, 2*nInliers, 1000, NULL, info, NULL, NULL,pdAddData); // with analytic Jacobian

	//DebugShowCostForHomographyRefinement(par, pdMeasurements, 9, nInliers, pdAddData);

	//printf("Initial Cost=%g Final Cost=%g\n", info[0], info[1]);
	//printf("# function evaluations =%f # Jacobian evaluations=%f\n", info[7], info[8]);
	//printf("# Iter= %d\n",ret);

	delete [] pdMeasurements;
	delete [] pdAddData;

	H.m_lpdEntries[0] = par[0];
	H.m_lpdEntries[1] = par[1];
	H.m_lpdEntries[2] = par[2];
	H.m_lpdEntries[3] = par[3];
	H.m_lpdEntries[4] = par[4];
	H.m_lpdEntries[5] = par[5];
	H.m_lpdEntries[6] = par[6];
	H.m_lpdEntries[7] = par[7];
	H.m_lpdEntries[8] = par[8];
}