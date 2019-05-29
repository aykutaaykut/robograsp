/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <common.hpp>
#include <pc_segmentation.hpp>

#include <pcl/common/time.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <utils_pcl_ros.hpp>

//call back function to copy clouds
inline void
OpenNIOrganizedMultiPlaneSegmentation::cloud_cb_ (const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  if (!viewer->wasStopped ())
  {
    cloud_mutex.lock ();
    prev_cloud = cloud;
    cloud_mutex.unlock ();
  }
}

// Set up the visualization window
boost::shared_ptr<pcl::visualization::PCLVisualizer>
cloudViewer (pcl::PointCloud<PointT>::ConstPtr cloud)
{
  boost::shared_ptr < pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color (cloud, 0, 255, 0);
  viewer->addPointCloud<PointT> (cloud, single_color, "cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.15, "cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  viewer->removeCoordinateSystem ();
  viewer->setShowFPS (false);
  return (viewer);
}

void
OpenNIOrganizedMultiPlaneSegmentation::removePreviousDataFromScreen (size_t prev_models_size)
{
  char name[1024];
  for (size_t i = 0; i < prev_models_size; i++)
  {
    sprintf (name, "normal_%zu", i);
    viewer->removeShape (name);

    sprintf (name, "plane_%02zu", i);
    viewer->removePointCloud (name);
  }
}

void
OpenNIOrganizedMultiPlaneSegmentation::removePreviousCLustersFromScreen (size_t num_clusters)
{
  for (size_t i = 0; i < num_clusters; i++)
  {
    sprintf (name, "cluster_%lu", i);
    viewer->removeShape (name);
  }
}

void
OpenNIOrganizedMultiPlaneSegmentation::removeBoundingBoxesAndArrows(size_t numBoundingBoxes)
{
  for(int bb = 0; bb < numBoundingBoxes; bb++)
  {
    sprintf(name, "cube %d", bb);
    viewer->removeShape(name);
    sprintf(name, "line %d-a", bb);
    viewer->removeShape(name);
    sprintf(name, "line %d-b", bb);
    viewer->removeShape(name);
    sprintf(name, "line %d-c", bb);
    viewer->removeShape(name);
  }
  viewer->removeShape("arrow");
}

// display clusters in with color
void
OpenNIOrganizedMultiPlaneSegmentation::displayEuclideanClusters (const pcl::PointCloud<PointT>::CloudVectorType &clusters, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  unsigned char red [6] = {255,   0,   0, 255, 255,   0};
  unsigned char grn [6] = {  0, 255,   0, 255,   0, 255};
  unsigned char blu [6] = {  0,   0, 255,   0, 255, 255};
  unsigned char color[3];

  for (size_t i = 0; i < clusters.size (); i++)
  {
    color[0] = red[i%6]; color[1] = grn[i%6]; color[2] = blu[i%6];
    displayEuclideanCluster (clusters[i], viewer, color, i);
  }
}

//display clusters given thier color
void
OpenNIOrganizedMultiPlaneSegmentation::displayEuclideanClusters (const pcl::PointCloud<PointT>::CloudVectorType &clusters, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, std::vector<ColorVec > &colors)
{
  unsigned char color[3];

  for (size_t i = 0; i < clusters.size (); i++)
  {
    color[0] = colors[i][0]; color[1] = colors[i][1]; color[2] = colors[i][2];
    displayEuclideanCluster (clusters[i], viewer, color, i);
  }
}

// display each cluster given its color
void
OpenNIOrganizedMultiPlaneSegmentation::displayEuclideanCluster (const pcl::PointCloud<PointT> &cluster, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, unsigned char color[], size_t clusterNum)
{
    sprintf (name, "cluster_%lu",clusterNum);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color0(boost::make_shared<pcl::PointCloud<PointT> >(cluster),color[0],color[1],color[2]);
    if (!viewer->updatePointCloud (boost::make_shared<pcl::PointCloud<PointT> >(cluster),color0,name))
      viewer->addPointCloud (boost::make_shared<pcl::PointCloud<PointT> >(cluster),color0,name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, name);
}

void
OpenNIOrganizedMultiPlaneSegmentation::displayRegion(std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > &regions, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, size_t regionIndex)
{
    pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);
    pcl::PlanarRegion<PointT> asd;

    if(regionIndex < 0)
    {
      for (size_t i = 0; i < regions.size (); i++)
      {
        Eigen::Vector3f centroid = regions[i].getCentroid ();
        Eigen::Vector4f model = regions[i].getCoefficients ();
        pcl::PointXYZ pt1 = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
        pcl::PointXYZ pt2 = pcl::PointXYZ (centroid[0] + (0.5f * model[0]),
                                           centroid[1] + (0.5f * model[1]),
                                           centroid[2] + (0.5f * model[2]));
        sprintf (name, "normal_%zu", i);
        viewer->addArrow (pt2, pt1, 1.0, 0, 0, false, name);
        contour->points = regions[i].getContour ();
        sprintf (name, "plane_%02zu", i);
        pcl::visualization::PointCloudColorHandlerCustom <PointT> color (contour, 255,125,125);//red[i], grn[i], blu[i]);
        viewer->addPointCloud (contour, color, name);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, name);
      }
    }
    else
    {
      Eigen::Vector3f centroid = regions[regionIndex].getCentroid ();
      Eigen::Vector4f model = regions[regionIndex].getCoefficients ();
      pcl::PointXYZ pt1 = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
      pcl::PointXYZ pt2 = pcl::PointXYZ (centroid[0] + (0.5f * model[0]),
                                         centroid[1] + (0.5f * model[1]),
                                         centroid[2] + (0.5f * model[2]));
      sprintf (name, "normal_%zu", regionIndex);
      viewer->addArrow (pt2, pt1, 1.0, 0, 0, false, name);
      contour->points = regions[regionIndex].getContour ();
      sprintf (name, "plane_%02zu", regionIndex);
      pcl::visualization::PointCloudColorHandlerCustom <PointT> color (contour, 255,62,62);//red[i], grn[i], blu[i]);
      viewer->addPointCloud (contour, color, name);
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, name);
    }
}

void
OpenNIOrganizedMultiPlaneSegmentation::displayNormalCloud(pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud)
{
  viewer->removePointCloud ("normals");
  viewer->addPointCloudNormals<PointT,pcl::Normal>(prev_cloud, normal_cloud, 10, 0.05f, "normals");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "normals");
}

void
OpenNIOrganizedMultiPlaneSegmentation::displayNormalCloud(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal_cloud)
  {
    viewer->removePointCloud ("normals");
    viewer->addPointCloudNormals<PointT,pcl::Normal>(cloud, normal_cloud, 10, 0.05f, "normals");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "normals");
  }

void
OpenNIOrganizedMultiPlaneSegmentation::displayBoundingBox(Box3D &box, int box_num)
{
    // bounding box
    pcl::ModelCoefficients cube;
    Box3D_2_MC(box,cube); // fill in the box values
    //THERE IS EIGEN ALIGNEDBOX IF YOU WANT EASIER FUNCTIONALITY
    sprintf(name, "cube %d", box_num);
    viewer->addCube(cube, name);
    Cube_2_Arrows(cube, viewer, box_num);
}

void
OpenNIOrganizedMultiPlaneSegmentation::displayBoundingBoxes(std::vector<Box3D> &fittedBoxes)
{
  for(int i = 0; i < fittedBoxes.size(); i++)
  {
    Box3D box = fittedBoxes[i];
    displayBoundingBox(fittedBoxes[i],i);
  }
}

void
OpenNIOrganizedMultiPlaneSegmentation::displayPlane(pcl::PointCloud<PointT>::Ptr contour)
{
  size_t bs = 0;
  sprintf(name,"plane_%02zu", bs);
  pcl::visualization::PointCloudColorHandlerCustom <PointT> color2 (contour, 255,62,62);//red[i], grn[i], blu[i]);
  viewer->addPointCloud (contour, color2, name);
}

bool
OpenNIOrganizedMultiPlaneSegmentation::computeHistogram (std::vector<double> const &data,  int const nbins,  std::vector<std::pair<double, double> > &histogram)
{
  //resizing the vector to nbins to store histogram;
  histogram.resize (nbins);

  //find min and max in the data
  double min = data[0], max = data[0];
  for (int i = 1; i < data.size (); i++)
  {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  //finding the size of each bins
  double size = (max - min) / nbins;

  //fill x values of each bins by bin center
  for (int i = 0; i < nbins; i++)
  {
	histogram[i].first = min + (size * i) + size / 2; //size/2 for the middle of the bins
    histogram[i].second = 0; //initializing the freq to zero
  }

  if(max == min)
  {
    return false;
  }

  //fill the freq for each data
  for (int i = 0; i < data.size (); i++)
  {
    int index = int (floor ((data[i] - min) / size));
    if (index == nbins) index = nbins - 1; //including right boundary
    if(index < 0)
    {
      std::cout << index << " " << size << " " << data[i] << " " << min; std::cout << std::endl;
    }
    else if(index >= nbins)
    	std::cout << index << std::endl;
    else
      histogram[index ].second++;
  }

  return true;
}

void
OpenNIOrganizedMultiPlaneSegmentation::setWorkingVolumeThresholds(float* a)
{
  for(int i = 0; i < 6; i++)
    threshs[i] = a[i];

}

void
OpenNIOrganizedMultiPlaneSegmentation::setWorkingVolumeThresholds(float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
  threshs[0] = minx;
  threshs[1] = maxx;
  threshs[2] = miny;
  threshs[3] = maxy;
  threshs[4] = minz;
  threshs[5] = maxz;

}


//allow for reading a file
void
OpenNIOrganizedMultiPlaneSegmentation::initSegmentation(int color_seg, float distance_thresh, float color_thresh)
{
  // configure normal estimation
  ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor (0.025f); //(0.01f);//0.03f
  ne.setNormalSmoothingSize (20.0f);//15.0f//20.0f

  // create a euclidean cluster comparator
switch (color_seg)
{
case 0:
  euclidean_cluster_comparator_ = pcl::MyEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr (new pcl::MyEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label> ());
  break;
case 1:
  euclidean_cluster_comparator_ = pcl::RGBEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr (new pcl::RGBEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label> ());
  break;
case 2:
default:
  euclidean_cluster_comparator_ = pcl::HueEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr (new pcl::HueEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label> ());
break;
}
  euclidean_cluster_comparator_->setDistanceThreshold (distance_thresh, false);
  /*float color_thresh;
  std::cout << "color thresh: ";
  std::cin >> color_thresh;
  euclidean_cluster_comparator_->setColorThreshold(color_thresh);*/
  euclidean_cluster_comparator_->setColorThreshold(color_thresh*color_thresh);

  // configure the multi plane segmentor
  mps.setMinInliers (10000); //(10000);
  mps.setAngularThreshold (0.017453 * 3.0);// ); //4.5// 3 degrees
  mps.setDistanceThreshold (0.01); //0.01 in meters
  mps.setMaximumCurvature(1000.005);//0.001

  //set up color segmentation
  tree = boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);

  reg.setDistanceThreshold (10); //10
  reg.setPointColorThreshold (25); //6
  reg.setRegionColorThreshold (20); //5
  reg.setMinClusterSize (200); //600
  reg.setSearchMethod (tree);

  // setup vfh (this is for feature extraction and not for segmentatio, consider moving
  vfh.setSearchMethod (tree);

  sor = pcl::StatisticalOutlierRemoval<PointT>::Ptr(new pcl::StatisticalOutlierRemoval<PointT> ());
  sor->setMeanK (10);
  sor->setStddevMulThresh (1.0);
  sor->setNegative (true);


}

//isFilter: aray of 6 denoting whether to threshold xmin,xmax,ymin,ymax,zmin,zmax and limits is the corresponding limits
//probably there is a better built in way of doing this
void
OpenNIOrganizedMultiPlaneSegmentation::threshXYZ(pcl::PointCloud<PointT>::Ptr out_cloud, bool isFilter[], float limits[])
{
  for (size_t i = 0; i < out_cloud->size(); i++)
  {
          if (out_cloud->points[i].x < limits[0] && isFilter[0])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].x > limits[1] && isFilter[1])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].y < limits[2] && isFilter[2])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].y > limits[3] && isFilter[3])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].z < limits[4] && isFilter[4])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].z > limits[5] && isFilter[5])
      out_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();

   //out_cloud->points[i].z = out_cloud->points[i].z - 1.0;
  }
}


void
OpenNIOrganizedMultiPlaneSegmentation::threshXYZ2(pcl::PointCloud<PointT>::Ptr out_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, bool isFilter[], float limits[])
{
  for (size_t i = 0; i < out_cloud->size(); i++)
  {
          if (out_cloud->points[i].x < limits[0] && isFilter[0])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].x > limits[1] && isFilter[1])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].y < limits[2] && isFilter[2])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].y > limits[3] && isFilter[3])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].z < limits[4] && isFilter[4])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
    else  if (out_cloud->points[i].z > limits[5] && isFilter[5])
      normal->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
  }
}

void
OpenNIOrganizedMultiPlaneSegmentation::threshColor(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices &indices, bool isFilter[], float limits[])
{
  for (size_t i = 0; i < cloud->size(); i++)
  {
          if (cloud->points[i].r > limits[0] && cloud->points[i].r < limits[1] && isFilter[0])
      indices.indices.push_back(i);
    else  if (cloud->points[i].g > limits[2] && cloud->points[i].g < limits[3] && isFilter[1])
      indices.indices.push_back(i);
    else  if (cloud->points[i].b > limits[4] && cloud->points[i].b < limits[5] && isFilter[2])
      indices.indices.push_back(i);
  }
}

std::vector<int> indices;
void OpenNIOrganizedMultiPlaneSegmentation::preProcPointCloud(pcl::PointCloud<PointT>::Ptr filtered_cloud, bool noiseFilter)
{
  bool isFilter[] = {true, true, true, true, true, true};
  //float threshs[] = {MIN_X_THRESH, MAX_X_THRESH, MIN_Y_THRESH, MAX_Y_THRESH, MIN_Z_THRESH, MAX_Z_THRESH};
  threshXYZ(filtered_cloud, isFilter, threshs);

  if(noiseFilter)
  {
	  sor->setInputCloud(filtered_cloud);
	  sor->filter(indices);//filter(*filtered_cloud);

	  for (size_t i = 0; i < indices.size(); i++)
	  {
		filtered_cloud->points[indices[i]].z = std::numeric_limits<float>::quiet_NaN();
	  }
  }
}

void
OpenNIOrganizedMultiPlaneSegmentation::planeExtract(
    pcl::PointCloud<PointT>::Ptr filtered_cloud,
    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > &regions,
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud,
    std::vector<pcl::ModelCoefficients> &model_coefficients,
    std::vector<pcl::PointIndices> &inlier_indices,
    pcl::PointCloud<pcl::Label>::Ptr labels,
    std::vector<pcl::PointIndices> &label_indices,
    std::vector<pcl::PointIndices> &boundary_indices,
    bool estimate_normals
    )
{
  regions.clear();

  if(estimate_normals)
  {
    // Estimate normals
    double normal_start = pcl::getTime ();
    ne.setInputCloud (filtered_cloud);
    ne.compute (*normal_cloud);
    double normal_end = pcl::getTime ();
    if(verbose)
      std::cout << "Normal Estimation took " << double (normal_end - normal_start) << std::endl;
  }

// Segment planes
  double plane_extract_start = pcl::getTime ();
  mps.setInputNormals (normal_cloud);
  mps.setInputCloud (filtered_cloud);
  mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
  double plane_extract_end = pcl::getTime ();
  if(verbose)
    std::cout << "Plane extraction took " << double (plane_extract_end - plane_extract_start) << std::endl;
}

float OpenNIOrganizedMultiPlaneSegmentation::getClosestPlaneModel(std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > &regions, int &closestIndex)
{
  // get the closest plane
  int nRegions = 0;
  float closestDepth = 10.0;//MAX_Z_THRESH;
  closestIndex = -1;

  for (size_t i = 0; i < regions.size (); i++)
  {
    Eigen::Vector3f centroid = regions[i].getCentroid ();
    Eigen::Vector4f model = regions[i].getCoefficients ();
    /*if (model[3] > MAX_Z_THRESH || model[3] < MIN_Z_THRESH)
    {
    	// skip planes that are too far or too close
        continue;
    }*/
    if (model[3] < closestDepth)
    {
      closestDepth = model[3];
      closestIndex = i;
    }
    nRegions++;
  }

  pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);

  return closestDepth;
}

void
OpenNIOrganizedMultiPlaneSegmentation::segmentPointCloud(pcl::PointCloud<PointT>::Ptr filtered_cloud,
                                                         pcl::PointCloud<PointT>::CloudVectorType &clusters,
                                                         std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals
                                                         )
{
  pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);
  segmentPointCloud(filtered_cloud, clusters, cluster_normals, contour);
}

void
OpenNIOrganizedMultiPlaneSegmentation::segmentPointCloud(pcl::PointCloud<PointT>::Ptr filtered_cloud,
                                                         pcl::PointCloud<PointT>::CloudVectorType &clusters,
                                                         std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals,
                                                         pcl::PointCloud<PointT>::Ptr &contour)
{
  Eigen::Vector4f used_plane_model;
  segmentPointCloud(filtered_cloud, clusters, cluster_normals, contour, used_plane_model);
}

void
OpenNIOrganizedMultiPlaneSegmentation::segmentPointCloud(pcl::PointCloud<PointT>::Ptr filtered_cloud,
                                                         pcl::PointCloud<PointT>::CloudVectorType &clusters,
                                                         std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals,
                                                         pcl::PointCloud<PointT>::Ptr &contour,
                                                         Eigen::Vector4f &used_plane_model
                                                         )
{
  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

  std::vector<pcl::ModelCoefficients> model_coefficients;
  std::vector<pcl::PointIndices> inlier_indices;
  pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices> label_indices;
  std::vector<pcl::PointIndices> boundary_indices;
  planeExtract(filtered_cloud, regions, normal_cloud,model_coefficients,inlier_indices,labels,label_indices,boundary_indices);

  unsigned int min_cluster_size = 30;

// Segment objects
  double segment_start = pcl::getTime ();
  std::vector<size_t> clusterIndices;
  if (regions.size () > 0)
  {
    Eigen::Vector3f used_plane_centroid;
    int closest_index;
    float closestDepth = getClosestPlaneModel(regions, closest_index);
    used_plane_centroid = regions[closest_index].getCentroid ();
    used_plane_model    = regions[closest_index].getCoefficients ();

    if(contour != NULL)
      contour->points = regions[closest_index].getContour();

    std::vector<bool> plane_labels;
    plane_labels.resize (label_indices.size (), false);
    for (size_t i = 0; i < label_indices.size (); i++)
      if (label_indices[i].indices.size () > MIN_PLANE_SIZE)
        plane_labels[i] = true;

    /*for (size_t k = 0; k < filtered_cloud->size(); k++) {
        filtered_cloud->points[k].r = rgb2hue(filtered_cloud->points[k].r,filtered_cloud->points[k].g,filtered_cloud->points[k].b); //carefrul this is float, the lhs is unsigned int
     }*/

    euclidean_cluster_comparator_->setInputCloud (filtered_cloud);
    euclidean_cluster_comparator_->setLabels (labels);
    euclidean_cluster_comparator_->setExcludeLabels (plane_labels);

    pcl::PointCloud<pcl::Label> euclidean_labels;
    std::vector<pcl::PointIndices> euclidean_label_indices;
    pcl::OrganizedConnectedComponentSegmentation<PointT,pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
    euclidean_segmentation.setInputCloud (filtered_cloud);
    euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

    for (size_t i = 0; i < euclidean_label_indices.size (); i++)
    {
      if (euclidean_label_indices[i].indices.size () > MIN_CLUSTER_SIZE)
      {
        pcl::PointCloud<PointT> cluster;
        pcl::PointCloud<pcl::Normal> cluster_normal;

        pcl::copyPointCloud (*filtered_cloud,euclidean_label_indices[i].indices,cluster);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(cluster, centroid);

        // skip clusters that are not on the viewpoint side of the table
	// this needs to be solved in a different way
        float tmp = used_plane_model[0]*centroid[0]+used_plane_model[1]*centroid[1]+used_plane_model[2]*centroid[2]+used_plane_model[3];
        /*if (tmp < 0)
          continue;*/

        if(cluster.size() < min_cluster_size)
          continue;

        clusters.push_back (cluster);
        clusterIndices.push_back(i);
        pcl::copyPointCloud (*normal_cloud, euclidean_label_indices[i], cluster_normal);
        cluster_normals.push_back(cluster_normal);
      }
    }
    if(verbose)
      PCL_INFO ("Got %d euclidean clusters!\n", clusters.size ());
  }
  else
  {
    std::cout << "No planar regions found!" << std::endl;
  };

  double segment_end = pcl::getTime();
  if(verbose)
    std::cout << "Segmentation took " << double (segment_end - segment_start) << std::endl;
}

void OpenNIOrganizedMultiPlaneSegmentation::segmentPointCloudGivenSavedCluster(pcl::PointCloud<PointT>::Ptr filtered_cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud,
    pcl::PointCloud<PointT>::CloudVectorType &clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals,
    std::vector<pcl::PointIndices> &euclidean_label_indices)
{
   // Segment objects
  double segment_start = pcl::getTime ();

  for (size_t i = 0; i < euclidean_label_indices.size (); i++)
  {
    if(verbose) {
      std::cout << euclidean_label_indices[i].indices.size (); std::cout << std::endl;}
    if (euclidean_label_indices[i].indices.size () > MIN_CLUSTER_SIZE)
    {

      pcl::PointCloud<PointT> cluster;
      pcl::PointCloud<pcl::Normal> cluster_normal;

      pcl::copyPointCloud (*filtered_cloud,euclidean_label_indices[i].indices,cluster);

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(cluster, centroid);

      clusters.push_back (cluster);

      pcl::copyPointCloud (*normal_cloud, euclidean_label_indices[i], cluster_normal);
      cluster_normals.push_back(cluster_normal);
    }
  }
  if(verbose)
    PCL_INFO ("Got %d euclidean clusters!\n", clusters.size ());

  double segment_end = pcl::getTime();
  if(verbose)
    std::cout << "Segmentation took " << double (segment_end - segment_start) << std::endl;
}

void OpenNIOrganizedMultiPlaneSegmentation::colorSegmentation(
    pcl::PointCloud<PointT>::CloudVectorType &clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals,
    pcl::PointCloud<PointT>::CloudVectorType &clusters_with_color,
    std::vector<pcl::PointCloud<pcl::Normal> > &normals_with_color)
{
  for(int i = 0; i < clusters.size(); i++)
  {
    std::vector <pcl::PointIndices> cluster_indices;
    reg.setInputCloud(clusters[i].makeShared());
    //reg.setIndices (indices);
    reg.extract (cluster_indices);

    for(int j = 0; j < cluster_indices.size(); j++)
    {
      pcl::PointCloud<PointT> color_cluster;
      pcl::copyPointCloud(clusters[i],cluster_indices[j],color_cluster);
      clusters_with_color.push_back(color_cluster);

      pcl::PointCloud<pcl::Normal> cluster_normal_color;
      pcl::copyPointCloud (cluster_normals[i], cluster_indices[j], cluster_normal_color);
      normals_with_color.push_back(cluster_normal_color);
    }
  }
}

void OpenNIOrganizedMultiPlaneSegmentation::featureExtraction(pcl::PointCloud<PointT>::CloudVectorType &clusters, std::vector<pcl::PointCloud<pcl::Normal> > &normals, std::vector<pc_cluster_features> &feats)
{
  std::vector<Box3D> fittedBoxes;
  std::vector<ColorVec > cluster_colors;
  featureExtraction(clusters,normals,feats,fittedBoxes,cluster_colors);
}

void OpenNIOrganizedMultiPlaneSegmentation::featureExtraction(
    pcl::PointCloud<PointT>::CloudVectorType &clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &normals,
    std::vector<pc_cluster_features> &feats,
    std::vector<Box3D> &fittedBoxes,
    std::vector<ColorVec > &cluster_colors)
{
  Eigen::Vector4f used_plane_model(0,0,0,0);
  featureExtraction(clusters,normals,feats,fittedBoxes,cluster_colors,used_plane_model);
}

void OpenNIOrganizedMultiPlaneSegmentation::featureExtraction(
    pcl::PointCloud<PointT>::CloudVectorType &clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &normals,
    std::vector<pc_cluster_features> &feats,
    std::vector<Box3D> &fittedBoxes,
    std::vector<ColorVec > &cluster_colors,
    Eigen::Vector4f &used_plane_model    )
{
  double start = pcl::getTime();
  double extract_start = pcl::getTime();
  pcl::PointCloud<PointT>::CloudVectorType used_clusters;
  //pcl::PointCloud<PointT>::CloudVectorType myProjectedClusters;

  cluster_colors.clear();
  fittedBoxes.clear();
  feats.clear();

  std::vector<int> indices;
  //plotter->clearPlots();

  for (int i = 0; i < clusters.size(); i++)
  {
    //double f_start = pcl::getTime ();
    Eigen::Vector4f centroid;
    pcl::PointCloud<PointT> cluster = clusters[i];
    pcl::compute3DCentroid(cluster, centroid);

    //remove points with nan normals? why would they occur?

  // add cluster to used_clusters
    used_clusters.push_back(cluster);
    indices.push_back(i);

  //Feature Extraction
    pc_cluster_features feat(cluster, used_plane_model);

    fittedBoxes.push_back(feat.aligned_bounding_box);
    cluster_colors.push_back(feat.color);

    vfh.setInputCloud (cluster.makeShared());
    vfh.setInputNormals (normals[i].makeShared());

    //this is new:
    //vfh.setNormalizeBins(true); //so that we don't have to on the matlab side
    //vfh.setNormalizeDistance(true);

    vfh.compute(feat.vfhs);//*vfhs);

    //double f_end = pcl::getTime ();

    feats.push_back(feat);
    //std::cout << "VFH calculation took " << double (f_end - f_start) << std::endl;
  }
  double extract_end = pcl::getTime();
  if(verbose)
    std::cout << "Feature extraction took " << double (extract_end - extract_start) << std::endl;
}

std::vector<int> BFS(std::vector<bool> &adjacency_matrix, int vertex_count, int given_vertex, std::vector<bool>  &is_used)
{
  bool *mark = new bool[vertex_count]();

  std::vector<int> merge_list;

  std::queue<int> q;
  q.push(given_vertex);
  merge_list.push_back(given_vertex);
  mark[given_vertex] = true;

  while (!q.empty())
  {
    int current = q.front(); q.pop();

    for (int i = 0; i < vertex_count; ++i)
    {
      if (adjacency_matrix[current*vertex_count + i] && !mark[i])
      {
          mark[i] = true;
          is_used[i] = true;
          q.push(i);
          merge_list.push_back(i);
      }
    }
  }

  delete [] mark;

  return merge_list;
}

void
OpenNIOrganizedMultiPlaneSegmentation::mergeClusters(
    pcl::PointCloud<PointT>::CloudVectorType &clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &cluster_normals,
    std::vector<pc_cluster_features> &feats,
    pcl::PointCloud<PointT>::CloudVectorType &merged_clusters,
    std::vector<pcl::PointCloud<pcl::Normal> > &merged_normals,
    double hue_thresh, double z_thresh, double euc_thresh)
{

  int num_clusters = feats.size();

  std::vector<bool> adjacency_matrix(num_clusters*num_clusters); // merge_matrix[i][j] = merge_matrix[i*num_clusters + j];

  bool checkForFullDist = euc_thresh > 0;

  for(int i = 0; i < num_clusters; i++)
  {
    for(int j = i+1; j < num_clusters; j++)
    {
      //std::cout << feats[i].hue << " " << feats[j].hue << " " << hueDiff(feats[i].hue, feats[j].hue) << std::endl;
      bool setTrue = false;
      // the below merges based on centroid distance, fyi
      if((hueDiff(feats[i].hue, feats[j].hue) < hue_thresh))
      {
    	  if(checkForFullDist)
    	  {
    		double d = 0;
    		double a= 0;
    		for(int k = 0; k < 3; k++)
    		{
    		  a = (feats[i].centroid[k] - feats[j].centroid[k]);
    		  d += a*a;
    		}
    		d = sqrt (d);
    		if(d < euc_thresh)
    		{
    		  setTrue = true;
    		}
    	  }
    	  else if(myAbs((float)(feats[i].centroid[2] - feats[j].centroid[2])) < z_thresh)
    	  {
    		setTrue = true;
    	  }

    	  if(setTrue)
    	  {
    		adjacency_matrix[i*num_clusters+j] = true;
    		adjacency_matrix[j*num_clusters+i] = true;
    	  }
      }
    }//std::cout << std::endl;
  }

  std::vector<bool> is_used(num_clusters+1);

  std::vector<std::vector <int> > merge_list_list;

  for(int i = 0; i < num_clusters; i++)
  {
      if(is_used[i])
        continue;

      merge_list_list.push_back(BFS(adjacency_matrix, num_clusters, i, is_used));
  }

  /*for(int i = 0; i < merge_list_list.size(); i++)
  {
    for(int j = 0; j < merge_list_list[i].size(); j++)
    {
      std::cout << merge_list_list[i][j] << " " ;
    } std::cout << std::endl;
  }*/


  for(int j = 0; j < merge_list_list.size(); j++)
  {
    //merged_cluster.clear();
    //merged_cluster_normal.clear();
    merged_clusters.push_back(clusters[merge_list_list[j][0]]);
    merged_normals.push_back(cluster_normals[merge_list_list[j][0]]);


    for(int i = 1; i < merge_list_list[j].size(); i++)
    {
      merged_clusters[j] += clusters[merge_list_list[j][i]];
      merged_normals[j]  += cluster_normals[merge_list_list[j][i]];
    }
    /*
    merged_cluster += (*used_clusters)[merge_list[1]];

    merged_cluster_normal += (*used_cluster_normals)[merge_list[1]];
    */
  }

}

//a lot of objects created and destroyed
int OpenNIOrganizedMultiPlaneSegmentation::processOnce (
		pcl::PointCloud<PointT>::ConstPtr prev_cl,
    pcl::PointCloud<PointT>::Ptr selected_cluster,
		std::vector<pc_cluster_features> &feats,
    std::vector<Box3D> &fittedBoxes,
		double main_hue, double hue_thresh, double z_thresh, double euc_thresh,
		bool preProc, int color_seg_ind, bool merge_clusters,
		bool displayAllBb, bool viewer_enabled, bool noiseFilter)
{

// misc variables, find a better way than static
  static size_t prev_models_size = 0;
  static size_t prev_cluster_num = 0;

  static int prev_boundingbox_num = 0;

  double outer_start = pcl::getTime();

  int color_index = -1;

  if (prev_cl)
  {
    double start = pcl::getTime();
    pcl::PointCloud<PointT>::Ptr filtered_prev_cloud(new pcl::PointCloud<PointT>(*prev_cl));
    cloud_mutex.unlock ();
    if(preProc)
      preProcPointCloud(filtered_prev_cloud, noiseFilter);


    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

    //is all of this needed?
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;

    pcl::PointCloud<PointT>::CloudVectorType clusters;
    std::vector<pcl::PointCloud<pcl::Normal> > cluster_normals;

    pcl::PointCloud<PointT>::CloudVectorType merged_clusters;
    std::vector<pcl::PointCloud<pcl::Normal> > merged_normals;

    pcl::PointCloud<PointT>::CloudVectorType *used_clusters;
    std::vector<pcl::PointCloud<pcl::Normal> > *used_cluster_normals;

    pcl::PointCloud<PointT>::CloudVectorType color_clusters;
    std::vector<pcl::PointCloud<pcl::Normal> > color_cluster_normals;

    // variables for the used plane contours
    pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);

    Eigen::Vector4f used_plane_model;

    double t1 = pcl::getTime();
    segmentPointCloud(filtered_prev_cloud, clusters, cluster_normals, contour, used_plane_model);

    double t2 = pcl::getTime();
    if(verbose)
      std::cout << "segment took " << double (t2 - t1) << std::endl;

    double extract_start = pcl::getTime();
    if(verbose)
      std::cout << "Preproc and spatial segmentation took " << double (extract_start - start) << std::endl;

    std::vector<int> indices;
    // std::vector<Box3D> fittedBoxes;
    std::vector<ColorVec > cluster_colors;

    //std::vector<pc_cluster_features> feats;

    if(clusters.empty())
    {
      if(viewer_enabled)
      {
        if (!viewer->updatePointCloud<PointT> (filtered_prev_cloud, "cloud"))
        {
            viewer->addPointCloud<PointT> (filtered_prev_cloud, "cloud");
        }
        viewer->removeShape ("plane_00");
        displayPlane(contour);

        // clear the visualizer
       removePreviousDataFromScreen (prev_models_size);
       removePreviousCLustersFromScreen(prev_cluster_num);
       removeBoundingBoxesAndArrows(prev_boundingbox_num);

      }
      return -1;
    }

    used_clusters = &clusters;
    used_cluster_normals = &cluster_normals;

    featureExtraction(*used_clusters, *used_cluster_normals, feats, fittedBoxes, cluster_colors, used_plane_model);

//merge
    if(merge_clusters)
    {
    	//std::cout <<"merging" << std::endl;
      mergeClusters(clusters, cluster_normals, feats, merged_clusters, merged_normals, hue_thresh, z_thresh, euc_thresh);
      used_clusters = &merged_clusters;
      used_cluster_normals = &merged_normals;
      featureExtraction(*used_clusters, *used_cluster_normals, feats, fittedBoxes, cluster_colors, used_plane_model);
      //if(verbose)
      //  std::cout << used_plane_model << std::endl;
    }

    //double main_hue = 60; //light blue 180, yellow 60
    double min_diff = 1000;
    std::vector<double> hue_diffs (feats.size());

    for(int i = 0; i < feats.size(); i++)
    {
      hue_diffs[i] = hueDiff(feats[i].hue, main_hue);//abs(feats[i].hue - main_hue);
      if(hue_diffs[i] < min_diff)
      {
        min_diff =hue_diffs[i];
        color_index = i;
      }
    }
    //
    if(verbose)
      std::cout << "Number of clusters of interest: " << used_clusters->size() << std::endl;

    double extract_end = pcl::getTime();
    if(verbose)
      std::cout << "Feature extraction and clusters of interest detection took " << double (extract_end - extract_start) << std::endl;

    if(viewer_enabled)
    {
      if (!viewer->updatePointCloud<PointT> (filtered_prev_cloud, "cloud"))
      {
        viewer->addPointCloud<PointT> (filtered_prev_cloud, "cloud");
      }

      // clear the visualizer
      removePreviousDataFromScreen (prev_models_size);
      removePreviousCLustersFromScreen(prev_cluster_num);
      removeBoundingBoxesAndArrows(prev_boundingbox_num);

      // Draw Visualization
      //displayRegion(regions,viewer, closestIndex);
      displayPlane(contour);
      displayEuclideanClusters (*used_clusters, viewer, cluster_colors);
      if(displayAllBb)
        displayBoundingBoxes(fittedBoxes);
      else
        displayBoundingBox(fittedBoxes[color_index],0);
    }

    prev_models_size = used_clusters->size();
    prev_cluster_num = used_clusters->size();
    prev_boundingbox_num = fittedBoxes.size();
    // selected_cluster = (*used_clusters)[color_index].makeShared();
    pcl::copyPointCloud((*used_clusters)[color_index], *selected_cluster);
    //cloud_mutex.unlock ();
    double end = pcl::getTime();
    if(verbose)
      std::cout << "Processing loop took " << double (end - start) << std::endl;
  }

  double outer_end = pcl::getTime();

  if(verbose)
    std::cout << "Overall loop took " << double (outer_end - outer_start) << std::endl << std::endl;

  return color_index;
}
