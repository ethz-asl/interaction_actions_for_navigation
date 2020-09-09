#include "opencv2/xfeatures2d.hpp"
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image window";

  //
  // now, you can no more create an instance on the 'stack', like in the tutorial
  // (yea, noticed for a fix/pr).
  // you will have to use cv::Ptr all the way down:
  int main(){
  //loading the images
  cv::Mat res;
  cv::Mat img_1 = cv::imread("/home/benjamin/pepper_ws/src/asl_pepper/asl_pepper_visual_localization/pictures_visual_loc/2.png", 3);
  if(! img_1.data )                              // Check for invalid input
  {
      cout <<  "Could not open or find the img_1" << std::endl ;
      return -1;
  }

  cv::Mat img_2 = cv::imread("/home/benjamin/pepper_ws/src/asl_pepper/asl_pepper_visual_localization/pictures_visual_loc/3.png", 3);
  if(! img_2.data )                              // Check for invalid input
  {
      cout <<  "Could not open or find the img_2" << std::endl ;
      return -1;
  }
  cout <<" before imshow" << endl;
  imshow("Img_1",img_1);
  waitKey(0);
  imshow("Img_2",img_2);
  waitKey(0);
  cout <<" imshow done" << endl;
  int nfeat, nOctavel;
  double contresh,edgethresh, sigma;
  nfeat = 0;
  nOctavel = 3;
  edgethresh = 10;
  sigma = 1.6;
  contresh = 0.1;

  // Choose the features SIFT or ORB
  cv::Ptr<cv::Feature2D> f2d = xfeatures2d::SIFT::create(nfeat,nOctavel,contresh,edgethresh,sigma);
  //cv::Ptr<ORB> f2d = cv::ORB::create();


  //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
  //cv::Ptr<Feature2D> f2d = ORB::create();
  // you get the picture, i hope..

  //-- Step 1: Detect the keypoints:
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  f2d->detect( img_1, keypoints_1 );
  f2d->detect( img_2, keypoints_2 );



  //-- Step 2: Calculate descriptors (feature vectors)
  Mat descriptors_1, descriptors_2;
  f2d->compute( img_1, keypoints_1, descriptors_1 );
  f2d->compute( img_2, keypoints_2, descriptors_2 );


  //BFMatcher matcher(NORM_HAMMING,1);
  //DescriptorMatcher::create(BruteForc);
  //std::vector< DMatch > matches;
  //matcher.match( descriptors_1, descriptors_2, matches,2 );

// nw try
  std::vector<std::vector<cv::DMatch>> matches;
  cv::BFMatcher matcher;
  matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches
  vector<cv::DMatch> good_matches;
  for (int i = 0; i < matches.size(); ++i)
  {
    const float ratio = 0.8; // As in Lowe's paper; can be tuned
    if (matches[i][0].distance < ratio * matches[i][1].distance)
    {
      good_matches.push_back(matches[i][0]);
    }
  }




  drawKeypoints(img_1, keypoints_1, img_1);
  imshow("igm_1 SIFT_features", img_1);
  waitKey(0);
  drawKeypoints(img_2, keypoints_2, img_2);
  imshow("igm_2 SIFT_features", img_2);
  waitKey(0);
  cout <<"avant drawMatches"<< endl;

  cout << "size of matches " << matches.size() << endl;
  cout << "size of keypoints_1 " << keypoints_1.size() << endl;
  cout << "size of keypoints_2 " << keypoints_2.size() << endl;
  
  //while( i < matches>size() ){
  //  cout << keypoints_1  << endl;
  //}
  //while(i < matches.size()){
  //  cout << matches
  //}



  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, res);
  imshow("matches",res);
  waitKey(0);

}
/*
  //-- Step 3: Matching descriptor vectors using BFMatcher :
  BFMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, res, -1,-1);
  imshow("matches",res);
  imwrite("matches.jpg",res);
}
*/
