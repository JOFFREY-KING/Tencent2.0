/*****************************

���ո�����Կ���Ч��

******************************/
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#define cut_rows 3   //�ֿ���
#define cut_cols 3   //�ֿ���

int main()
{
	Mat srcImg=imread("1.jpg");
	imshow("src",srcImg);

	vector<Mat> ceilImg;//����и�ͼ������

	int height=srcImg.rows;//ԭͼ�ĸ�
	int width=srcImg.cols;//ԭͼ�Ŀ�

	//(cut_rows-1)*(cut_cols-1)��ͼ����СӦ����һ���ģ����ұ�һ�к����±�һ��ͼ����С���ܲ�һ��
	int ceil_height=(int)(height/cut_rows);
	int ceil_width=(int)(width/cut_cols);
	int ceil_down_height=height-(cut_rows-1)*ceil_height;//���±�ͼ���ĸ�
	int ceil_right_width=width-(cut_cols-1)*ceil_width;//���ұ�ͼ���Ŀ�

	for(int i=0;i<cut_rows-1;i++)    //���������±�һ��
		for(int j=0;j<cut_cols;j++)
		{
			if(j<cut_cols-1)
			{
				Rect rect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				ceilImg.push_back(srcImg(rect));
			
			}
			else   //���ұ�һ�У����������½ǣ�
			{
				Rect rect((cut_cols-1)*ceil_width,i*ceil_height,ceil_right_width,ceil_height);
				ceilImg.push_back(srcImg(rect));
			}
		}

	for(int i=0;i<cut_cols;i++)//���±�һ�У��������½ǣ�������һ�ж������
	{
		if(i<cut_cols-1)
		{
			Rect rect(i*ceil_width,(cut_rows-1)*ceil_height,ceil_width,ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
		else   //���½����ͼ���
		{
			Rect rect((cut_cols-1)*ceil_width,(cut_rows-1)*ceil_height,ceil_right_width,ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
	}

	cout<<"�ֿ������"<<ceilImg.size()<<endl;
	Mat dst;
	for(int i=0;i<ceilImg.size();i++)
	{
		dst=ceilImg[i];
		imshow("dst",dst); 
		waitKey(0);
	}

	waitKey(0);
	return 0;
}
