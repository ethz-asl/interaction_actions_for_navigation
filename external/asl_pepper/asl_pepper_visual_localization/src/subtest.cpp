// ros opencv interface librairies
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>

//OpenCV librairies
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/core.hpp>


// basic C++ librairies
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <chrono> // to set the low frequency of the pipeline
#include <thread> // same as above

using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window";

struct data_storage {Mat image; int image_id; vector<KeyPoint> keypoints;  Mat descriptors;};


class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  image_transport::Publisher res_pub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/pepper_robot/camera/front/image_raw", 1,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    res_pub_ = it_.advertise("/image_converter/output_matching",1);

    cv::namedWindow(OPENCV_WINDOW);

  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

private:
  // Parameter of the sift features
  const int nfeat = 0; const int nOctavel = 3 ;
  const double edgethresh = 10; const double sigma = 1.6; const double contresh = 0.1;
  // variables in order to store data
  int i=0;
  const double treshold = 15;
  vector< data_storage > database;
  std::string str_i; std::string str_j; std::string str_tot;
//  vector<KeyPoint> * keypoints_n(nullptr); // pointer of keypoints_n
//  keypoints_n = new std::vector<KeyPoint>;



  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Image processing here
    // Choose the features SIFT or ORB
    cv::Ptr<cv::Feature2D> f2d = xfeatures2d::SIFT::create(nfeat,nOctavel,contresh,edgethresh,sigma);
    //cv::Ptr<ORB> f2d = cv::ORB::create();

    std::vector<KeyPoint> keypoints_n;
    f2d->detect( cv_ptr->image, keypoints_n );
    Mat descriptors_n;
    f2d->compute( cv_ptr->image, keypoints_n, descriptors_n );
    drawKeypoints(cv_ptr->image, keypoints_n, cv_ptr->image);

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(30);
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());

   // store the data
    database.push_back(data_storage());
    database[i].image = cv_ptr->image;
    database[i].image_id = i;
    database[i].keypoints = keypoints_n;
    database[i].descriptors = descriptors_n;


    // Find 2 similar images
    int j;
    for ( j = 0; j < i; j++) {
    std::vector<std::vector<cv::DMatch>> matches;
    cv::BFMatcher matcher;
    matcher.knnMatch(database[i].descriptors, database[j].descriptors, matches, 2);  // Find two nearest matches
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i)
    {
      const float ratio = 0.5; // As in Lowe's paper; can be tuned ; original ratio = 0.8;
      if (matches[i][0].distance < ratio * matches[i][1].distance)
      {
        good_matches.push_back(matches[i][0]);
      }
    }

     if (good_matches.size() >= (treshold/100.)*database[i].keypoints.size() || good_matches.size() >= (treshold/100.)*database[j].keypoints.size()) {
      str_i = std::to_string(i); str_j = std::to_string(j);
      str_tot = ("match between image " + str_i + " and image " + str_j + " ! ");
      Mat res;
      drawMatches(database[i].image, database[i].keypoints, database[j].image, database[j].keypoints, good_matches, res);
      imshow("it's a match !",res);
      waitKey(30);
      sensor_msgs::ImagePtr pub_match = cv_bridge::CvImage(std_msgs::Header(), "bgr8", res).toImageMsg();
      res_pub_.publish(pub_match);
      cout << str_tot << '\n';
      break;
    }
   }


   std::cout << "voici le compte :  " << i << '\n';
   i = i +1 ;
   this_thread::sleep_for (std::chrono::seconds(15));

    // wait in order to have a low frequency pipeline
   //std::this_thread::sleep_for (std::chrono::seconds(3));

  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter voilavoila;
  ros::spin();
  return 0;
}
