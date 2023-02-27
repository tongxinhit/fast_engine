// ROS 库
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

// MMDelpoy 库
#include "mmdeploy/detector.h"
 
using namespace std;

// 初始化相机内参，调用camera_info_callback获取相机内参
bool is_K_empty = 1;
double K[9];

// 全局变量：图像矩阵和点云
cv_bridge::CvImagePtr color_ptr, depth_ptr;
cv::Mat color_pic, depth_pic;

// 用于获取相机的RGB图像，并转换为cv::Mat 形式
void color_Callback(const sensor_msgs::ImageConstPtr& color_msg)
{
//   cv_bridge::CvImagePtr color_ptr;
  try
  {
    // 这部分可以可视化看看是否成功获得了图像
    // cv::imshow("color_view", cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)->image);
    color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);    
    // cv::waitKey(1050); // 不断刷新图像，频率时间为int delay，单位为ms
  }
  catch (cv_bridge::Exception& e )
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
  }
  
  color_pic = color_ptr->image;
// 可视化 用于查看RGB图像尺寸。
//   cout<<"output some info about the rgb image in cv format"<<endl;
//   cout<<"rows of the rgb image = "<<color_pic.rows<<endl; 
//   cout<<"cols of the rgb image = "<<color_pic.cols<<endl; 
//   cout<<"type of rgb_pic's element = "<<color_pic.type()<<endl; 
}

// 获取深度图信息，实现参考RGB图像
void depth_Callback(const sensor_msgs::ImageConstPtr& depth_msg)
{
//   cv_bridge::CvImagePtr depth_ptr;
  try
  { 
    // 此处注意 depth 图像的数据格式有两种。
    // unsigned short *depth_data = (unsigned short*)&depth_msg->data[0];
    // cv::imshow("depth_view", cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image);
    depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1); 
    // cv::imshow("depth_view", cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image);
    // depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1); 

    // cv::waitKey(1050);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
  }

  depth_pic = depth_ptr->image;

  // output some info about the depth image in cv format
//   cout<<"output some info about the depth image in cv format"<<endl;
//   cout<<"rows of the depth image = "<<depth_pic.rows<<endl; 
//   cout<<"cols of the depth image = "<<depth_pic.cols<<endl; 
//   cout<<"type of depth_pic's element = "<<depth_pic.type()<<endl; 
}

// 用于获取相机内参
void camera_info_callback(const sensor_msgs::CameraInfoConstPtr &camera_info_msg)
{
    // 读取相机参数
    if(is_K_empty)
    {
        for(int i=0; i<9; i++)
        {
            K[i] = camera_info_msg->K[i];
        }
        is_K_empty = 0;
    }
}


int main(int argc, char **argv)
{
//   初始化节点
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
//   定义两个opencv窗口 用于对color_Callback和depth_Callback回调函数可视化
  cv::namedWindow("color_view");
  cv::namedWindow("depth_view");
  cv::startWindowThread();

// image_transport订阅节点信息，定义发布信息的节点。
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, color_Callback);
//   image_transport::Subscriber sub1 = it.subscribe("/camera/aligned_depth_to_color/image_raw", 1, depth_Callback);
  ros::Subscriber sub_cmara_info = nh.subscribe("/camera/depth/camera_info", 1, camera_info_callback);
//   ros::Publisher pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("generated_pc", 1);
  image_transport::Publisher img_publisher = it.advertise("generated_pc", 1);

// 开始创建mmdeploy，初始化一些参数。
  auto device_name = "cuda";
  auto model_path = "/home/tongxin/mmdeploy/work_dir/yolox_tiny";
  mmdeploy_detection_t* bboxes{};
  int* res_count{};
  mmdeploy_detector_t detector{};
  int status{};
//   创建detection模型
  status = mmdeploy_detector_create_by_path(model_path, device_name, 0, &detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
    return 1;
  }

  double sample_rate = 4; // 4HZ，1秒发4次 
  ros::Rate naptime(sample_rate); // use to regulate loop rate 

  cout<<"Start detect"<<endl;

  while (ros::ok()) {
    // 当彩色图为空时，订阅图像信息。
    if (!color_pic.data) {
        fprintf(stdout, "failed to load image\n");
        ros::spinOnce(); //allow data update from callback; 
        naptime.sleep(); 
        continue;
    }
    // 创建可检测的数据格式
    mmdeploy_mat_t mat{
    color_pic.data, color_pic.rows, color_pic.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};
    // 创建用于保存检测结果的boxes
    mmdeploy_detection_t* bboxes{};
    int* res_count{};
    status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
    if (status != MMDEPLOY_SUCCESS) {
      fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
      return 1;
    }
    // 打印检测框的个数
    fprintf(stdout, "bbox_count=%d\n", *res_count);
    // 打印检测框的信息
    for (int i = 0; i < *res_count; ++i) {
      const auto& box = bboxes[i].bbox;
      const auto& mask = bboxes[i].mask;

      fprintf(stdout, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
              i, box.left, box.top, box.right, box.bottom, bboxes[i].label_id, bboxes[i].score);

      // skip detections with invalid bbox size (bbox height or width < 1)
      if ((box.right - box.left) < 1 || (box.bottom - box.top) < 1) {
        continue;
      }
    
      // skip detections less than specified score threshold
      if (bboxes[i].score < 0.3) {
        continue;
      }

      cv::rectangle(color_pic, cv::Point{(int)box.left, (int)box.top},
                    cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_pic).toImageMsg();

    // 设置发布节点信息并发布。
    img_publisher.publish(msg);
    
    ros::spinOnce(); //allow data update from callback; 
    naptime.sleep(); // wait for remainder of specified period; 
  }
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  mmdeploy_detector_destroy(detector);
    
  cv::destroyWindow("color_view");
  cv::destroyWindow("depth_view");
}