/*
 * def_loop.cpp
 *
 *  Created on: Sep 4, 2014
 *      Author: baris
 *  Updated on: Feb 23, 2018 :)))
 */

#include <pc_segmentation.hpp>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/openni_grabber.h>
#include <pcl_ros/point_cloud.h>

#include <utils.hpp>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <utils_pcl_ros.hpp>
#include <geometry_msgs/Transform.h>

#include <k2g.h>

#include <signal.h>

typedef pcl::PointCloud<pcl::PointXYZ> MyPointCloud;

pcl::PointCloud<PointT>::ConstPtr prev_cloud;
pcl::PointCloud<PointT>::Ptr selected_cluster (new pcl::PointCloud<PointT>);

boost::mutex cloud_mutex;
boost::mutex imageName_mutex;
bool writePCD2File = false;
char imageName[100];
bool gotFirst = false;
bool interrupt = false;

//./def_loop -src 1 -rt asus_filtered -dt 0.03 -ct 15 -t 14 -e 0.1 -v 40 -b 0
//./def_loop -src 1 -rt asus_filtered -dt 0.03 -ct 15 -t 14 -e 0.1 -v 40 -b 0 -sh 0
//./def_loop -src 1 -rt asus_filtered -dt 0.03 -ct 15 -t 14 -e 0.1 -v 200 -sh 1 -st 0.2 -sv 200
//./def_loop -dt 0.03 -ct 15 -t 14 -e 0.1 -v 340 -sh 1 -st 0.2 -sv 240
//./def_loop -dt 0.03 -ct 15 -t 14 -e 0.1 -v 200 -sh 1 -st 0.4 -sv 20


void
cloud_cb_ros_ (const sensor_msgs::PointCloud2ConstPtr& msg)
{
  pcl::PCLPointCloud2 pcl_pc;

  if (msg->height == 1){
    sensor_msgs::PointCloud2 new_cloud_msg;
    new_cloud_msg.header = msg->header;
    new_cloud_msg.height = 480;
    new_cloud_msg.width = 640;
    new_cloud_msg.fields = msg->fields;
    new_cloud_msg.is_bigendian = msg->is_bigendian;
    new_cloud_msg.point_step = msg->point_step;
    new_cloud_msg.row_step = 20480;
    new_cloud_msg.data = msg->data;
    new_cloud_msg.is_dense = msg->is_dense;

    pcl_conversions::toPCL(new_cloud_msg, pcl_pc);
  }
  else
    pcl_conversions::toPCL(*msg, pcl_pc);

  pcl::PointCloud<PointT> cloud;
  pcl::fromPCLPointCloud2(pcl_pc, cloud);
  //std::cout << cloud.height << std::endl;
  //pcl::PCLPointCloud2 asd = *msg;

  //pcl::fromPCLPointCloud2(asd, *cloud);

  cloud_mutex.lock ();
  prev_cloud = cloud.makeShared();
  if(writePCD2File)
  {
    pcl::PointCloud<PointT>::Ptr saved_cloud(new pcl::PointCloud<PointT>(*prev_cloud));
    std::cout << imageName << std::endl;
    cloud_mutex.unlock ();
    imageName_mutex.lock();
    pcl::io::savePCDFile(imageName, *saved_cloud);
    imageName_mutex.unlock();
    writePCD2File = false;
  }
  else
    cloud_mutex.unlock ();

  gotFirst = true;
}

void interruptFn(int sig)
{
  interrupt = true;
}

void
cloud_cb_direct_ (const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  cloud_mutex.lock ();
  prev_cloud = cloud;
  if(writePCD2File)
  {
    pcl::PointCloud<PointT>::Ptr saved_cloud(new pcl::PointCloud<PointT>(*prev_cloud));
    std::cout << imageName << std::endl;
    cloud_mutex.unlock ();
    imageName_mutex.lock();
    pcl::io::savePCDFile(imageName, *saved_cloud);
    imageName_mutex.unlock();
    writePCD2File = false;
  }
  else
    cloud_mutex.unlock ();

  gotFirst = true;
}

inline void
fake_cloud_cb_kinectv2_ (const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  prev_cloud = cloud;
  if(writePCD2File)
  {
    pcl::PointCloud<PointT>::Ptr saved_cloud(new pcl::PointCloud<PointT>(*prev_cloud));
    std::cout << imageName << std::endl;
    cloud_mutex.unlock ();
    imageName_mutex.lock();
    pcl::io::savePCDFile(imageName, *saved_cloud);
    imageName_mutex.unlock();
    writePCD2File = false;
  }
}

int
main (int argc, char **argv)
{

  parsedArguments pA;
  if(parseArguments(argc, argv, pA) < 0)
    return 0;

  ridiculous_global_variables::ignore_low_sat       = pA.saturation_hack;
  ridiculous_global_variables::saturation_threshold = pA.saturation_hack_value;
  ridiculous_global_variables::saturation_mapped_value = pA.saturation_mapped_value;

  OpenNIOrganizedMultiPlaneSegmentation multi_plane_app;
  multi_plane_app.verbose = pA.verbose;

  std::vector<Box3D> fittedBoxes;
  pcl::PointCloud<PointT>::Ptr cloud_ptr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr ncloud_ptr (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Label>::Ptr label_ptr (new pcl::PointCloud<pcl::Label>);

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  multi_plane_app.initSegmentation(pA.seg_color_ind, pA.ecc_dist_thresh, pA.ecc_color_thresh);

  if(pA.viz)
  {
    viewer = cloudViewer(cloud_ptr);
    multi_plane_app.setViewer(viewer);
  }

  float selected_object_features[324];

  float workSpace[] = {-0.6,0.6,-0.5,0.5,0.3,2.0};//Simon on the other side:{-0.1,0.6,-0.4,0.15,0.7,1.1};//{-0.5,0.6,-0.4,0.4,0.4,1.1}; //TODO: What is this?
  multi_plane_app.setWorkingVolumeThresholds(workSpace);

  pcl::Grabber* interface;
  ros::NodeHandle *nh;
  ros::Subscriber sub;
  ros::Publisher pub;

  ros::Publisher transformPub;

  bool spawnObject = true;
  ros::Publisher objMarkerPub;

  ros::Publisher segmentedPub;

  ros::Publisher bbPub;

  K2G *k2g;
  processor freenectprocessor = OPENGL;

  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud;

  if(pA.ros_node)
  {
    std::cout << "ros node initialized" << std::endl;
    ros::init(argc, argv, "pc_segmentation",ros::init_options::NoSigintHandler);
    nh = new ros::NodeHandle();
  }

  const char *outRostopic = "/baris/features";
  const char *outRostopicTransform = "/baris/objectTransform";
  const char *outRostopicSegmented = "/baris/segmentedPC";
  const char *outRostopicObjects = "/baris/objects";

  if(pA.output_type == comType::cROS)
  {
    std::cout << "Publishing ros topic: " << outRostopic << std::endl;
    pub = nh->advertise<pc_segmentation::PcFeatures>(outRostopic,5);

    transformPub = nh->advertise<geometry_msgs::Transform>(outRostopicTransform,5);
    segmentedPub = nh->advertise<pc_segmentation::PcSegmented>(outRostopicSegmented,5);
    bbPub = nh->advertise<pc_segmentation::PcObjects>(outRostopicObjects, 5);

    if(spawnObject)
    {
      objMarkerPub = nh->advertise<visualization_msgs::Marker>("/baris/object_marker", 1);
    }
  }

  switch (pA.pc_source)
  {
    case 0:
    {
      std::cout << "Using ros topic as input" << std::endl;
      sub = nh->subscribe<sensor_msgs::PointCloud2>(pA.ros_topic, 1, cloud_cb_ros_);
      break;
    }
    case 1:
    {
      std::cout << "Using the openni device" << std::endl;

      interface = new pcl::OpenNIGrabber ();

      boost::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> f = boost::bind (&cloud_cb_direct_,_1);
      boost::signals2::connection c = interface->registerCallback (f);

      interface->start ();
      break;
    }
    case 2:
    default:
    {
      std::cout << "Using kinect v2" << std::endl;

      freenectprocessor = static_cast<processor>(pA.freenectProcessor);

      k2g = new K2G(freenectprocessor, true);
      cloud = k2g->getCloud();
      prev_cloud = cloud;
      gotFirst = true;
      break;
    }
  }

  /*ofstream myfile;
  myfile.open ("centroids.txt");*/
  if(!pA.viz)
   signal(SIGINT, interruptFn);
  while (!interrupt && (!pA.viz || !viewer->wasStopped ()))
  {
    if(pA.ros_node)
    {
      ros::spinOnce();
    }
    if(pA.viz)
      viewer->spinOnce(20);
    if(!gotFirst)
      continue;
    std::vector<pc_cluster_features> feats;
    int selected_cluster_index = -1;
    if(cloud_mutex.try_lock ())
    {
      if(pA.pc_source == 2)
      {
        cloud = k2g->getCloud();
        fake_cloud_cb_kinectv2_(cloud);
      }

  	  if(pA.viz && pA.justViewPointCloud)
  	  {
  		pcl::PointCloud<PointT>::Ptr filtered_prev_cloud(new pcl::PointCloud<PointT>(*prev_cloud));
  		multi_plane_app.preProcPointCloud(filtered_prev_cloud);
  		if (!viewer->updatePointCloud<PointT> (filtered_prev_cloud, "cloud"))
          {
              viewer->addPointCloud<PointT> (filtered_prev_cloud, "cloud");
          }
          selected_cluster_index = -1;
  	  }
  	  else
  	  {
  		  selected_cluster_index = multi_plane_app.processOnce(prev_cloud, selected_cluster, feats, fittedBoxes,
          pA.hue_val,pA.hue_thresh, pA.z_thresh, pA.euc_thresh, pA.pre_proc,
          pA.seg_color_ind, pA.merge_clusters, pA.displayAllBb, pA.viz, pA.filterNoise); //true is for the viewer

  	  }

      cloud_mutex.unlock();

      /*the z_thresh may result in wrong cluster to be selected. it might be a good idea to do
       * some sort of mean shift tracking, or selecting the biggest amongst the candidates (vs choosing the most similar color)
       * or sending all the plausible ones to c6 and decide there
       */
    }

    if(selected_cluster_index < 0)
      continue;

    if(pA.verbose){
      float angle = feats[selected_cluster_index].aligned_bounding_box.angle;
      std::cout << "Selected cluster angle (rad, deg): " << angle << " " << angle*180.0/3.14159 << std::endl;
      std::cout << "Selected cluster hue: " << feats[selected_cluster_index].hue << std::endl;
    }

    //send the object features to c6 here
    //fillObjectInfo(outFeatures);

    // VPFH Features Ros Msg
    pc_segmentation::PcFeatures rosMsg;
    pc_cluster_features selected_features = feats[selected_cluster_index];
    fillRosMessage(rosMsg, selected_features);
    pub.publish(rosMsg);
    transformPub.publish(rosMsg.transform);

    // Segmented Object Ros MSg
    pc_segmentation::PcSegmented segmentedRosMsg;
    // Fill segmentation message TODO: Convert this to a function
    segmentedRosMsg.header.stamp = ros::Time::now();
    segmentedRosMsg.bb_center.x = selected_features.aligned_bounding_box.center.x;
    segmentedRosMsg.bb_center.y = selected_features.aligned_bounding_box.center.y;
    segmentedRosMsg.bb_center.z = selected_features.aligned_bounding_box.center.z;
    segmentedRosMsg.transform.translation.x = selected_features.aligned_bounding_box.center.x;
    segmentedRosMsg.transform.translation.y = selected_features.aligned_bounding_box.center.y;
    segmentedRosMsg.transform.translation.z = selected_features.aligned_bounding_box.center.z;
    segmentedRosMsg.transform.rotation.x = selected_features.aligned_bounding_box.rot_quat[0];
    segmentedRosMsg.transform.rotation.y = selected_features.aligned_bounding_box.rot_quat[1];
    segmentedRosMsg.transform.rotation.z = selected_features.aligned_bounding_box.rot_quat[2];
    segmentedRosMsg.transform.rotation.w = selected_features.aligned_bounding_box.rot_quat[3];
    sensor_msgs::PointCloud2 pc2_msg;
    toROSMsg(*selected_cluster, pc2_msg);
    segmentedRosMsg.segmented_pc = pc2_msg;
    segmentedPub.publish(segmentedRosMsg);
    // End of segmentation msg

    // All bounding boxes Ros Msg
    pc_segmentation::PcObjects objectsRosMsg;
    fillObjectsRosMessage(objectsRosMsg, feats);
    bbPub.publish(objectsRosMsg);

    objectPoseTF(rosMsg.transform);
    if(spawnObject)
    {
      visualization_msgs::Marker marker;
      getObjectMarker(marker, rosMsg);
      objMarkerPub.publish(marker);
    }

  }
  if(pA.pc_source == 1)
  {
    interface->stop ();
    delete interface;
  }
  if(pA.ros_node)
  {
    delete nh;

    //myfile.close();
    ros::shutdown();
  }
  return 0;
}
