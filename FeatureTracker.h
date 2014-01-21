#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Utils.h"

using namespace std;

class FeatureTracker {

public:
    FeatureTracker(vector3i dim);
   ~FeatureTracker();

    void ExtractAllFeatures();

    // Set seed at current time step. FindNewFeature will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
    void FindNewFeature(vector3i seed);

    // Track forward based on the center points of the features at the last time step
    void TrackFeature(float* pData, int direction, int mode);
    void SaveExtractedFeatures(int index)       { featureSequence_[index] = currentFeatures_; }
    void SetDataPtr(float* pData)               { data_.assign(pData, pData+volumeSize_); }
    void SetTFRes(int res)                      { tfRes_ = res; }
    void SetTFMap(float* map)                   { tfMap_.assign(map, map+tfRes_); }
    float* GetMaskPtr()                         { return mask_.data(); }
    int GetTFResolution()                       { return tfRes_; }
    int GetVoxelIndex(const vector3i &v)        { return blockDim_.x*blockDim_.y*v.z+blockDim_.x*v.y+v.x; }

    // Get all features information of current time step
    vector<Feature>* GetFeatureVectorPointer(int index) { return &featureSequence_[index]; }

private:
    vector3i predictRegion(int index, int direction, int mode); // Predict region t based on direction, returns offset
    void fillRegion(Feature& f, const vector3i& offset);        // Scanline algorithm - fills everything inside edge
    void expandRegion(Feature& f);                              // Grows edge where possible
    void shrinkRegion(Feature& f);                              // Shrinks edge where nescessary
    bool expandEdge(Feature& f, const vector3i& seed);          // Sub-func inside expandRegion
    void shrinkEdge(Feature& f, const vector3i& seed);          // Sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);                      // Update the feature vectors information after tracking

    float getOpacity(float value) { return tfMap_[(int)(value * (tfRes_-1))]; }

    vector<float> data_;        // Raw volume intensity value
    vector<float> mask_;        // Mask volume, same size with a time step data
    vector<float> maskPrev_;    // Mask volume, same size with a time step data
    vector<float> tfMap_;       // Tranfer function setting

    float globalMaskValue_ = 0.0f;  // Global mask value for newly detected features
    int tfRes_ = 1024;              // Default transfer function resolution
    int volumeSize_;
    int timeLeft2Forward_;
    int timeLeft2Backward_;

    vector3i blockDim_;

    vector<Feature> currentFeatures_; // Features info in current time step
    vector<Feature> backup1Features_; // ... in the 1st backup time step
    vector<Feature> backup2Features_; // ... in the 2nd backup time step
    vector<Feature> backup3Features_; // ... in the 3rd backup time step

    std::unordered_map<int, vector<Feature> > featureSequence_;
};

#endif // FEATURETRACKER_H
