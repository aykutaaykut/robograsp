/*
 * offlineProc.cpp
 *
 *  Created on: Sep 9, 2014
 *    Author: baris
 */
#include<pc_segmentation.hpp>
#include<utils_pcl_ros.hpp>

int starting_subject = 0;
int starting_demo = 0;
int starting_kf = 0;
int starting_skill = 0;
bool is_set_z_thresh = false;
bool is_viz_only = false;
bool batch = false;

void readInputArgs(int argc, char *argv[])
{
  for(int i =1; i < argc; i++)
  {
    if(!strcmp(argv[i],"-on"))
    {
      starting_subject = atoi(argv[++i]);
    }
    else if(!strcmp(argv[i],"-os"))
    {
      starting_skill = atoi(argv[++i]);
    }
    else if(!strcmp(argv[i],"-od"))
    {
      starting_demo = atoi(argv[++i]);
    }
    else if(!strcmp(argv[i],"-ok"))
    {
      starting_kf = atoi(argv[++i]);
    }
    else if(!strcmp(argv[i],"-ov"))
    {
      i++;
      is_viz_only = true;
    }
    else if(!strcmp(argv[i],"-batch"))
    {
        i++;
        batch = true;
    }
  }
}

//THE BATCH PORTION IS HACKZ ALL THE WAY, IF YOU WANT TO DO A BETTER JOB FOLLOW THE RABBIT AND FIX THAT SHIT

//./offlineProcV2 -dt 0.03 -ct 15 -t 14 -e 0.1 -v 200 -b 0 -sh 1 -sv 200

int main(int argc, char *argv[])
{
  readInputArgs(argc, argv);
  parsedArguments pA;
  parseArguments(argc,argv,pA);

  ridiculous_global_variables::ignore_low_sat       = pA.saturation_hack;
  ridiculous_global_variables::saturation_threshold = pA.saturation_hack_value;
  ridiculous_global_variables::saturation_mapped_value = pA.saturation_mapped_value;

  bool only_color = false;
  bool no_color = !only_color && true;

  OpenNIOrganizedMultiPlaneSegmentation segmenter;
  char folder_path[] ="/home/ceteke/Documents/datasets/ipek";  //"/home/baris/data/agLearning";
  char subject_base[] = "";//"subject";
  char folder_base[] = "pc";
  char *skill_names[] = {"redbox"};//{"open", "test_open"};//{"do something", "close the box","pour"};//{"test_close the box","test_pour"};//{"close the box", "insert", "place"};
  char demo_base[] = "demo";
  char kf_base[] = "kf";
  char separator[] = "_";
  char ext[] = "pcd";
  char processed_folder[] = "proc";
  char name[1024];

  /*char folder_path[] = "./";
  char skill_name[] = "iter";
  char object_name[] = "cluster";
  char separator[] = "_";
  char name[1024];*/

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  pcl::PointCloud<PointT>::Ptr cloud_ptr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr ncloud_ptr (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Label>::Ptr label_ptr (new pcl::PointCloud<pcl::Label>);

  viewer = cloudViewer(cloud_ptr);

  int keyPress;
  bool jumpOut = false;
  bool give_info = false;
  bool load_pc = true;
  bool process_pc = false;
  bool viz_proc_result = false;
  bool viz_ind_clusters = true;
  bool save_features = false;
  bool isObs = false;
  bool isCls = false;
  bool is_select_clusters = false;
  bool save_selected_cluster = false;
  bool save_euclidean_indices = false;
  bool massive_save = false;
  bool save_separate = false;
  bool separate_white = false;
  bool auto_merge = true;
  int selectedCluster = 0;
  int subject_num = starting_subject-1;//0;
  int skill_num = starting_skill;
  int demo_num = starting_demo;
  int kf_num = starting_kf;

  bool has_subjects = strcmp(subject_base,"");

  int cluster_num = 0;

  std::vector<int> selected_clusters;

  int subject_list[] = {1,2,3,4,5,6,7,8,14};//,9,10};

  segmenter.initSegmentation(pA.seg_color_ind, pA.ecc_dist_thresh, pA.ecc_color_thresh);
  segmenter.setViewer(viewer);
  float threshs[] = {-0.2,0.2,-0.3,0.1,0.6,1.1};
  segmenter.setWorkingVolumeThresholds(threshs);

  size_t prev_models_size = 0;
  size_t prev_cluster_num = 0;
  int prev_boundingbox_num = 0;

  pcl::PointCloud<PointT>::CloudVectorType *used_clusters;
  std::vector<pcl::PointCloud<pcl::Normal> > *used_cluster_normals;
  std::vector<pc_cluster_features> feats;
  std::vector<Box3D> fittedBoxes;
  std::vector<ColorVec > cluster_colors;
  pcl::PointCloud<PointT>::CloudVectorType clusters;
  std::vector<pcl::PointCloud<pcl::Normal> > cluster_normals;
  pcl::PointCloud<PointT>::CloudVectorType color_clusters;
  std::vector<pcl::PointCloud<pcl::Normal> > color_cluster_normals;
  pcl::PointCloud<PointT>::CloudVectorType merged_clusters;
  std::vector<pcl::PointCloud<pcl::Normal> > merged_normals;

  //int merge_list[2];
  std::vector<int> merge_list;
  std::vector<std::vector<int> > merge_list_list;
  bool is_merge = false;
  pcl::PointCloud<PointT> merged_cluster;
  pcl::PointCloud<pcl::Normal> merged_cluster_normal;

  pcl::PointCloud<PointT> saved_cluster_xyzrgba;
  pcl::PointCloud<pcl::Normal> saved_cluster_normal;
  pcl::PointCloud<pcl::PointXYZRGBNormal> saved_cluster_all;

  std::vector<pcl::PointIndices> label_indices;

  std::stringstream base_skill_path;
  char bsp[1024];

  float pct = 0;
  float rct = 0;

  double color_thresh = pA.ecc_color_thresh;

  //from visualization
  segmenter.removeCoordinateSystem();

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (50);

    if(batch)
    {
      load_pc = true;
      process_pc = true;
      save_features = true;
    }
    else
    {
      //min bytes to read, min time to wait in 10^{-1} seconds,
      keyPress = nonBlockingWait(0,1);
      if(keyPress != EOF)
      {
        std::cout << std::endl;
      }
      switch (keyPress)
      {
        case 27: //esc
          jumpOut = true;
          break;
        case 'n': //next subject
          subject_num++;
          demo_num = 0;
          kf_num = 0;
          load_pc = true;
          break;
        case 'b': //previous subject
          subject_num--; if(subject_num < 0) subject_num = 0;
          demo_num = 0;
          kf_num = 0;
          load_pc = true;
          break;
        case 'w':
          skill_num++;
          demo_num = 0;
          kf_num = 0;
          load_pc = true;
          break;
        case 's':
          skill_num--; if(skill_num < 0) skill_num = 0;
          kf_num = 0;
          demo_num = 0;
          load_pc = true;
          break;
        case 'a':
          demo_num--; if(demo_num < 0) demo_num = 0;
          kf_num = 0;
          load_pc = true;
          break;
        case 'd':
          demo_num++;
          kf_num = 0;
          load_pc = true;
          break;
        case 'k':
          kf_num++;
          load_pc = true;
          break;
        case 'j':
          kf_num--; if(kf_num < 0) kf_num = 0;
          load_pc = true;
          break;
        case 'i':
          give_info = true;
          break;
        case 'l':
          load_pc = true;
          break;
        case 'p':
          process_pc = true;
          break;
        case 'v':
          viz_proc_result = true;
          break;
        case 'r':
          subject_num = 0;
          skill_num = 0;
          demo_num = 0;
          kf_num = 0;
          merge_list.clear();
          merge_list_list.clear();
          selected_clusters.clear();
          break;
        case 'c':
          viz_ind_clusters = !viz_ind_clusters;
          break;
        case 'x':
          selectedCluster++;
          std::cout << "Current cluster: " << selectedCluster << std::endl;
          break;
        case 'z':
          selectedCluster--; if(selectedCluster < 0) selectedCluster = 0;
          std::cout << "Current cluster: " << selectedCluster << std::endl;
          break;
        case 'o':
          isObs = !isObs;
          std::cout << "Observation: " << isObs << std::endl;
          break;
        case 'y':
          isCls = !isCls;
          std::cout << "Using Clusters: " << isCls << std::endl; //using saved indices
          break;
        case 't':
        {
          is_select_clusters = !is_select_clusters;
          massive_save = is_select_clusters;
          save_separate = is_select_clusters;
          if(!is_select_clusters)
            break;
          bool flag = true;
          selected_clusters.clear();
          int tmp;
          while (flag)
          {
            std::cout << "Cluster " << selected_clusters.size()+1 << ": ";
            std::cin >> tmp;

            if(tmp < 0)
              flag = false;
            else
              selected_clusters.push_back(tmp);
          }
          std::cout << "Selected Clusters List: " << selected_clusters.size()  << std::endl;
          break;
        }
        case 'm':
        {
          is_merge = !is_merge;
          if(!is_merge)
            break;
          bool flag = true;
          bool new_list = false;
          merge_list.clear();
          merge_list_list.clear();
          int tmp;
          while(flag)
          {
            if(new_list)
            {
              merge_list.clear();
              new_list = false;
            }

            while (!new_list)
            {
              std::cout << "Cluster " << merge_list.size()+1 << ": ";
              std::cin >> tmp;

              if (tmp == -9)
              {
                new_list = true;
                break;
              }
              else if(tmp < 0)
              {
                flag = false;
                break;
              }
              else
                merge_list.push_back(tmp);
            }
            std::cout << "Merge List: " << merge_list.size()  << std::endl;
            merge_list_list.push_back(merge_list);
          }
          break;
        }
        case 'q':
          //no_color = !no_color;
          if(color_thresh == 10000)
            color_thresh = pA.ecc_color_thresh;
          else
            color_thresh = 10000;
          segmenter.euclidean_cluster_comparator_->setColorThreshold(color_thresh);
          break;
        case 'f':
          if(!viz_ind_clusters)
          {
            std::cout << "First press c to enter individual cluster mode." << std::endl;
            break;
          }
          //cout << "Saving features" << endl;
          save_features= true;
          break;
        case 'e': //automerge on
          auto_merge = !auto_merge;
          break;
        case 'g':
          /*if(!viz_ind_clusters)
          {
            cout << "First press c to enter individual cluster mode." << endl;
            break;
          }*/
          //cout << "Saving selected cluster" << endl;
          //save_selected_cluster = true;
          save_euclidean_indices = true;
          cluster_num++;
          break;
        case '+':
          pA.hue_val++;
          process_pc = true;
          break;
        case '-':
          pA.hue_val--;
          process_pc = true;
          break;
        case '=':
          std::cin >> pct;
          std::cin >> rct;
          segmenter.reg.setPointColorThreshold(pct);
          segmenter.reg.setRegionColorThreshold(pct);
          break;
        case ']':
          pA.saturation_mapped_value++;
          process_pc = true;
          break;
        case '[':
          pA.saturation_mapped_value--;
          process_pc = true;
          break;
        case 'h':
          system("cat commands.txt");
          break;
        case 'u':
          is_set_z_thresh = !is_set_z_thresh;
          break;
        case '0':
          separate_white = true;
          break;
        default:
        break;
      }

      if(jumpOut)
        break;

      if(give_info)
      {
        std::cout << "Subject: " << subject_list[subject_num] << std::endl;
        std::cout << "Skill: "   << skill_names[skill_num] << std::endl;
        std::cout << "Demo: "    << demo_num << std::endl;
        std::cout << "KF: "      << kf_num << std::endl;
        std::cout << "Region Thresh: " << segmenter.reg.getRegionColorThreshold() << " Point Thresh: " << segmenter.reg.getPointColorThreshold() << std::endl;
        std::cout << "Observation: " << isObs << std::endl;

        if(viz_ind_clusters)
        {
            std::cout << "Selected Cluster: " << selectedCluster << std::endl;
            std::cout << "Pose: " << feats[selectedCluster].centroid[0] << " " << feats[selectedCluster].centroid[1] << " " << feats[selectedCluster].centroid[2] << " " << feats[selectedCluster].bb_orientation;
            std::cout << std::endl;
            std::cout << "Color: " << feats[selectedCluster].color[0] << " " << feats[selectedCluster].color[1] << " " << feats[selectedCluster].color[2] << " " << feats[selectedCluster].hue;
            std::cout << std::endl;
            std::cout << "Size: " << feats[selectedCluster].volume2;
            std::cout << std::endl;
        }
        if(merge_list_list.size() > 0)
        {
          for(int j = 0; j < merge_list_list.size(); j++)
          {
            std::cout << "Merge List: " << j << std::endl;
            for(int i = 0; i < merge_list_list[j].size(); i++)
              std::cout << merge_list_list[j][i] << " ";
            std::cout << std::endl;
          }
        }
        if(selected_clusters.size() > 0)
        {
          std::cout << "Selected Clusters List: " << std::endl;
          for(int i = 0; i < selected_clusters.size(); i++)
            std::cout << selected_clusters[i] << " ";
          std::cout << std::endl;
        }
        give_info = false;
      }
    }

    if(load_pc)
    {
      cluster_num = 0;//a bad idea to do it here!

      base_skill_path.clear();
      base_skill_path.str(std::string());
      if(has_subjects)//(subject_base[0] != "")
        base_skill_path << folder_path << "/" << subject_base << subject_list[subject_num] << "/" << skill_names[skill_num];
      else
      {
        base_skill_path << folder_path << "/" << skill_names[skill_num] << "/" << folder_base;
      }
      std::cout << base_skill_path.str() << std::endl;
      sprintf(bsp,"%s",base_skill_path.str().c_str());

      sprintf(name,"%s/%s",bsp,processed_folder);
      makeFolder(name);

      //sprintf(name,"%s%d.%s",folder_path,subject_base,subject_num,skill_names[skill_num],demo_base,demo_num,separator,kf_base,kf_num,ext);
      std::stringstream file_path;

      file_path << bsp << "/";
      if(isCls)
      {
        file_path << processed_folder << "/";
      }
      if(isObs)
      {
          file_path << "observation" << separator;
      }
      file_path <<  demo_base << demo_num << separator << kf_base << kf_num;

      if(isCls)
      {
          std::stringstream file_path_normals;

          file_path_normals << file_path.str() << separator << "cluster_normals." << ext;
          file_path << separator << "clusters." << ext;

          sprintf(name,"%s",file_path.str().c_str());
          if(!loadPCD(name,cloud_ptr))
          {
            load_pc = false;
            continue;
          }

          sprintf(name,"%s",file_path_normals.str().c_str());
          if(!loadPCD(name,ncloud_ptr))
          {
            load_pc = false;
            continue;
          }

          if(isObs)
            sprintf(name,"%s/%s/observation_demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);
          else
            sprintf(name,"%s/%s/demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);

          label_indices.clear();
          std::vector< std::vector<int> > tmp_inds;
          file2doubleVector(tmp_inds,name);
          for(int i=0; i<tmp_inds.size();i++)
          {
              pcl::PointIndices indices;
              indices.indices = tmp_inds[i];
              label_indices.push_back(indices);
          }

          /*pcl::PointIndices indices;
          //label_indices
          file2vector(indices.indices,name);
          label_indices.push_back(indices);*/
      }
      else
      {
        file_path << "." << ext;

        sprintf(name,"%s",file_path.str().c_str());
        if(!loadPCD(name,cloud_ptr))
        {
          load_pc = false;

          //below code is not debugged fully
          if(batch)
          {
            if(kf_num == 0)
            {
              if(demo_num == 0)
              {
                exit_batch:
                std::cout << "Congratulations! You have processed all the files. " << std::endl;
                break;
              }
              else if(has_subjects)
              {
                std::cout << "Finished subject: " << subject_num << std::endl;
                subject_num++;
                demo_num = 0;
                kf_num = 0;
              }
              else
              {
                goto exit_batch;
              }
            }
            else
            {
              std::cout << "Finished demo: " << demo_num << std::endl;
              demo_num++;
              kf_num = 0;
            }
          }
          continue;
        }
      }
      load_pc = false;
      process_pc = true;// && !isCls;
    }
    if(is_viz_only)
    {
      if (!viewer->updatePointCloud<PointT> (cloud_ptr, "cloud"))
        viewer->addPointCloud<PointT> (cloud_ptr, "cloud");
      continue;
    }

    //Make proc and viz one shot
    pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>(*cloud_ptr));
    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Label>::Ptr label_cloud(new pcl::PointCloud<pcl::Label>);

    if(process_pc)
    {
      feats.clear();
      fittedBoxes.clear();
      cluster_colors.clear();
      clusters.clear();
      cluster_normals.clear();
      color_clusters.clear();
      color_cluster_normals.clear();

      if(!isCls)
      {
        //segmenter.preProcPointCloud(filtered_cloud);
        bool isFilter[] = {true,true,true,true,true,true};
        //float threshs[]  = {-0.5,0.6,-0.4,0.4,0.4,1.1};//{-0.4,0.4,-0.2, 0.2, 0.9, 1.4};
        if(is_set_z_thresh)
        {
          std::cin >> threshs[4];
          std::cin >> threshs[5];
            is_set_z_thresh = false;
        }
        segmenter.threshXYZ(filtered_cloud, isFilter, threshs);
      }

      if(only_color)
      {
        if(!isCls)
        {
          std::vector<pcl::ModelCoefficients> model_coefficients;
          std::vector<pcl::PointIndices> inlier_indices;
          pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
          std::vector<pcl::PointIndices> label_indices_tmp;
          std::vector<pcl::PointIndices> boundary_indices;
          segmenter.planeExtract(filtered_cloud, regions, normal_cloud,model_coefficients,inlier_indices,labels,label_indices_tmp,boundary_indices);
          int closest_index;
          float closestDepth = segmenter.getClosestPlaneModel(regions, closest_index);
          bool isFilter[] = {false,false,false,false,false,true};
          float threshs[]  = {0,0,0, 0, 0, closestDepth*0.98};
          segmenter.threshXYZ(filtered_cloud, isFilter, threshs);
          segmenter.threshXYZ2(filtered_cloud, normal_cloud, isFilter, threshs); //thresh normals with the cloud
          //remove nans from the clouds!!
          std::vector< int > index;
          pcl::removeNaNFromPointCloud(*filtered_cloud,*filtered_cloud,index);
          pcl::removeNaNNormalsFromPointCloud(*normal_cloud,*normal_cloud,index);
          clusters.push_back(*filtered_cloud);
          cluster_normals.push_back(*normal_cloud);
        }
        else
        {
            clusters.push_back(*cloud_ptr);
            cluster_normals.push_back(*ncloud_ptr);
        }
      }
      else
      {
        if(!isCls)
          segmenter.segmentPointCloud(filtered_cloud, clusters, cluster_normals);
        else
          segmenter.segmentPointCloudGivenSavedCluster(cloud_ptr, ncloud_ptr, clusters, cluster_normals, label_indices);
      }

      if(no_color)
      {
        used_clusters = &clusters;
        used_cluster_normals = &cluster_normals;
      }
      else
      {
        segmenter.colorSegmentation(clusters, cluster_normals, color_clusters, color_cluster_normals);
        //segmenter.colorSegmentationCC(clusters, cluster_normals, color_clusters, color_cluster_normals);
        used_clusters = &color_clusters;
        used_cluster_normals = &color_cluster_normals;
      }

      if(is_merge && !auto_merge)
      {
        for(int j = 0; j < merge_list_list.size(); j++)
        {
          merged_cluster.clear();
          merged_cluster_normal.clear();

          merged_cluster  = (*used_clusters)[merge_list_list[j][0]];
          merged_cluster_normal  = (*used_cluster_normals)[merge_list_list[j][0]];

          for(int i = 1; i < merge_list_list[j].size(); i++)
          {
            merged_cluster        += (*used_clusters)[merge_list_list[j][i]];
            merged_cluster_normal += (*used_cluster_normals)[merge_list_list[j][i]];
          }
          /*
          merged_cluster += (*used_clusters)[merge_list[1]];

          merged_cluster_normal += (*used_cluster_normals)[merge_list[1]];
          */

          used_clusters->push_back(merged_cluster);
          used_cluster_normals->push_back(merged_cluster_normal);
        }

        is_merge = false;
      }

      if(separate_white)
      {
        pcl::PointCloud<PointT> color_cluster;
        pcl::PointCloud<pcl::Normal> color_normal;
        pcl::PointIndices pc_inds;
        bool isFilter[] = {true,true,true};
        float tLow;
        float tHigh;
        cin >> tLow;
        cin >> tHigh;
        //float threshs[]  = {249.0,253.0,249.0, 253.0, 249.0, 253.0};
        float threshs[]  = {tLow,tHigh,tLow,tHigh,tLow,tHigh};
        segmenter.threshColor((*used_clusters)[selectedCluster].makeShared(), pc_inds, isFilter, threshs);

        pcl::copyPointCloud((*used_clusters)[selectedCluster],pc_inds,color_cluster);
        pcl::copyPointCloud((*used_cluster_normals)[selectedCluster],pc_inds,color_normal);

        used_clusters->push_back(color_cluster);
        used_cluster_normals->push_back(color_normal);

        separate_white = false;
      }

      segmenter.featureExtraction(*used_clusters, *used_cluster_normals, feats, fittedBoxes, cluster_colors);

      if(auto_merge)
      {
        merged_clusters.clear();
        merged_normals.clear();
        segmenter.mergeClusters(*used_clusters, *used_cluster_normals, feats, merged_clusters, merged_normals, pA.hue_thresh, pA.z_thresh, pA.euc_thresh);
        used_clusters = &merged_clusters;
        used_cluster_normals = &merged_normals;
        segmenter.featureExtraction(*used_clusters, *used_cluster_normals, feats, fittedBoxes, cluster_colors);
        selectedCluster = findFeatVecByHue(feats,pA.hue_val);
        std::cout << "Auto merge selected cluster: " << selectedCluster << std::endl;
        //auto_merge = false;
      }

      std::cout << "Number of clusters of interest: " << used_clusters->size() << std::endl;
      for(int i = 0; i<used_clusters->size(); i++)
        std::cout << (*used_clusters)[i].size() << "\t";
      std::cout << std::endl;

      process_pc= false;
      viz_proc_result = true;
    }

    /*if(isCls)
    {
      if (!viewer->updatePointCloud<PointT> (cloud_ptr, "cloud"))
        viewer->addPointCloud<PointT> (cloud_ptr, "cloud");
    }*/

    if(viz_proc_result)
    {
      if (!viewer->updatePointCloud<PointT> (filtered_cloud, "cloud"))
        viewer->addPointCloud<PointT> (filtered_cloud, "cloud");
      // clear the visualizer
      segmenter.removePreviousDataFromScreen (prev_models_size);
      segmenter.removePreviousCLustersFromScreen(prev_cluster_num);
      segmenter.removeBoundingBoxesAndArrows(prev_boundingbox_num);
      // Draw Visualization
      //displayRegion(regions,viewer, closestIndex);
      //displayPlane(contour);
      segmenter.displayEuclideanClusters (*used_clusters, viewer, cluster_colors);
      segmenter.displayBoundingBoxes(fittedBoxes);
      //pcl::PointCloud<pcl::Normal>::Ptr tmp = normals_with_color[0].makeShared();
      //displayNormalCloud(used_clusters[0].makeShared(),normals_with_color[0].makeShared());
      prev_models_size = used_clusters->size();
      prev_cluster_num = used_clusters->size();
      prev_boundingbox_num = fittedBoxes.size();
      viz_proc_result = false;
    }

    if(viz_ind_clusters)
    {
      //selectedCluster =  getClusterByColor(feats, 2);

      if(selectedCluster >= (*used_clusters).size())
        selectedCluster = (*used_clusters).size()-1;

      bool isFilter[] = {true,true,true,true,true,true};
      segmenter.threshXYZ(filtered_cloud, isFilter, threshs);

      if (!viewer->updatePointCloud<PointT> (filtered_cloud, "cloud"))
        viewer->addPointCloud<PointT> (filtered_cloud, "cloud");

      segmenter.removePreviousDataFromScreen (prev_models_size);
      segmenter.removePreviousCLustersFromScreen(prev_cluster_num);
      segmenter.removeBoundingBoxesAndArrows(prev_boundingbox_num);

      unsigned char color[3];
      color[0] = cluster_colors[selectedCluster][0]; color[1] = cluster_colors[selectedCluster][1]; color[2] = cluster_colors[selectedCluster][2];
      segmenter.displayEuclideanCluster ((*used_clusters)[selectedCluster], viewer, color, 0); //(used_clusters[selectedCluster], viewer, cluster_colors);
      segmenter.displayBoundingBox(fittedBoxes[selectedCluster], 0);

      prev_models_size = 1;
      prev_cluster_num = 1;
      prev_boundingbox_num = 1;
    }

    if(save_euclidean_indices)//save_selected_cluster)
    {
      std::cout << "Saving selected clusters to a file" << std::endl;

      if(massive_save && !save_separate)
      {
        saved_cluster_xyzrgba.clear();
        saved_cluster_normal.clear();
        saved_cluster_all.clear();

        saved_cluster_xyzrgba = (*used_clusters)[selected_clusters[0]];
        saved_cluster_normal = (*used_cluster_normals)[selected_clusters[0]];

        std::vector< std::vector<int> > cur_indices_vec;
        cur_indices_vec.resize(selected_clusters.size());

        fillInIndices(cur_indices_vec[0],0,(*used_clusters)[selected_clusters[0]].size());

        //vector2file(cur_indices,name,",");

        int pre_count;
        int count = (*used_clusters)[selected_clusters[0]].size();

        for(int i = 1; i < selected_clusters.size(); i++ )
        {

          saved_cluster_xyzrgba += (*used_clusters)[selected_clusters[i]];
          saved_cluster_normal  += (*used_cluster_normals)[selected_clusters[i]];

          pre_count = count;
          count+=(*used_clusters)[selected_clusters[i]].size();

          fillInIndices(cur_indices_vec[i],pre_count,count);

        }
        //pcl::concatenateFields(saved_cluster_xyzrgba, saved_cluster_normal, saved_cluster_all);

        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_clusters.pcd",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_clusters.pcd",bsp,processed_folder,demo_num,kf_num);
        pcl::io::savePCDFile(name, saved_cluster_xyzrgba);

        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_cluster_normals.pcd",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_cluster_normals.pcd",bsp,processed_folder,demo_num,kf_num);
        pcl::io::savePCDFile(name, saved_cluster_normal);

        //SAVE INDICES
        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);

        doubleVector2file(cur_indices_vec,name,",","\n");

        //save here
      }
      else if(save_separate)
      {

        std::vector< std::vector<int> > cur_indices_vec;
        cur_indices_vec.resize(selected_clusters.size());

        int pre_count = 0;
        int count = (*used_clusters)[selected_clusters[0]].size();

        for(int i = 0; i < selected_clusters.size(); i++ )
        {
          //std::cout << "Saving cluster " << selected_clusters[i] << " with " << (*used_clusters)[selected_clusters[0]]
          fillInIndices(cur_indices_vec[i],pre_count,count);
          if(isObs)
            sprintf(name,"%s/%s/observation_demo%d_kf%d_clusters%d.pcd",bsp,processed_folder,demo_num,kf_num,i);
          else
            sprintf(name,"%s/%s/demo%d_kf%d_clusters%d.pcd",bsp,processed_folder,demo_num,kf_num,i);
          pcl::io::savePCDFile(name, (*used_clusters)[selected_clusters[i]]);

          if(isObs)
            sprintf(name,"%s/%s/observation_demo%d_kf%d_cluster_normals%d.pcd",bsp,processed_folder,demo_num,kf_num,i);
          else
            sprintf(name,"%s/%s/demo%d_kf%d_cluster_normals%d.pcd",bsp,processed_folder,demo_num,kf_num,i);
          pcl::io::savePCDFile(name, (*used_cluster_normals)[selected_clusters[i]]);

          //SAVE INDICES
          if(isObs)
            sprintf(name,"%s/%s/observation_demo%d_kf%d_indices%d.txt",bsp,processed_folder,demo_num,kf_num,i);
          else
            sprintf(name,"%s/%s/demo%d_kf%d_indices%d.txt",bsp,processed_folder,demo_num,kf_num,i);

          vector2file(cur_indices_vec[i],name,",");
        }



        //save here
      }
      else
      {
        saved_cluster_all.clear();
        //pcl::concatenateFields((*used_clusters)[selectedCluster], (*used_cluster_normals)[selectedCluster], saved_cluster_all);
        //pcl::io::savePCDFile(name, saved_cluster_all);

        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_clusters.pcd",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_clusters.pcd",bsp,processed_folder,demo_num,kf_num);
        pcl::io::savePCDFile(name, (*used_clusters)[selectedCluster]);

        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_cluster_normals.pcd",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_cluster_normals.pcd",bsp,processed_folder,demo_num,kf_num);
        pcl::io::savePCDFile(name, (*used_cluster_normals)[selectedCluster]);

        if(isObs)
          sprintf(name,"%s/%s/observation_demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);
        else
          sprintf(name,"%s/%s/demo%d_kf%d_indices.txt",bsp,processed_folder,demo_num,kf_num);

        std::vector<int> cur_indices;
        cur_indices.resize( (*used_clusters)[selectedCluster].size() );
        fillInIndices(cur_indices);
        vector2file(cur_indices,name,",");
      }
      save_selected_cluster = false;
      save_euclidean_indices = false;
      //save_features = true;
    }


    if(save_features) //should find a better way to do this portion
    {
      std::cout << "Saving features selected clusters to a csv file";
      if(isObs)
        sprintf(name,"%s/%s/observation_demo%d_kf%d.csv",bsp,processed_folder,demo_num,kf_num);
      else
        sprintf(name,"%s/%s/demo%d_kf%d.csv",bsp,processed_folder,demo_num,kf_num);

      if(massive_save && !save_separate)
      {
        if(isPath(name) == PT_FILE) //since we appending
          remove(name);
        std::cout << " for " << selected_clusters.size() << " number of clusters";
        for(int i = 0; i < selected_clusters.size(); i++)
          feats[selected_clusters[i]].write2file(name, true); //append write
      }
      else if(save_separate)
      {
        for(int i = 0; i < selected_clusters.size(); i++)
        {
          if(isObs)
            sprintf(name,"%s/%s/observation_demo%d_kf%d_cluster%d.csv",bsp,processed_folder,demo_num,kf_num,i);
          else
            sprintf(name,"%s/%s/demo%d_kf%d_cluster%d.csv",bsp,processed_folder,demo_num,kf_num,i);
          feats[selected_clusters[i]].write2file(name);
        }
      }
      else
        feats[selectedCluster].write2file(name); //overwrite
      std::cout << name << std::endl;
      save_features = false;
    }

    if(batch)
    {
        kf_num++;
    }

  }

  return 0;
}
