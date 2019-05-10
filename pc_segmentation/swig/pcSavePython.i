/* File : pythonIRCP.i */
%module pcSavePython

%include "carrays.i"
%array_class(float, floatArray);
 
#pragma SWIG nowarn=SWIGWARN_TYPEMAP_SWIGTYPELEAK

%mutable;

/* Some helper functions to make it easier to test */
%inline %{
	extern void init(bool _use_ros = true, char *ros_topic = "/asus/depth_registered/points",unsigned int sleep_milisec = 10);
  extern void save(const char *baseName);
	extern void fini();
  extern bool isSaving();
  
  extern void initFeats(char *_argv = NULL);
  extern bool getCurrentFeatures(float *out_feats, bool wait_for_availability = true, bool save_pc = false);
  extern bool getFileFeatures(float *out_feats, char *file_name);
  extern int getNumberOfFeatures();
%}
