#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Utils.h"

class FeatureTracker {

public:
    FeatureTracker(vec3i dim);
   ~FeatureTracker();

    void ExtractAllFeatures();

    // Set seed at current time step. FindNewFeature will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
    void FindNewFeature(vec3i seed);

    // Track forward based on the center points of the features at the last time step
    void TrackFeature(float* pData, int direction, int mode);
    void SaveExtractedFeatures(int index) { featureSequence[index] = currentFeatures; }
    void SetDataPtr(float* pData)         { data.assign(pData, pData+volumeSize); }
    void SetTFRes(int res)                { tfRes = res; }
    void SetTFMap(float* map)             { tfMap.assign(map, map+tfRes); }
    float* GetMaskPtr()                   { return mask.data(); }
    int GetTFResolution()                 { return tfRes; }
    int GetVoxelIndex(const vec3i& v)     { return blockDim.x*blockDim.y*v.z+blockDim.x*v.y+v.x; }

    // Get all features information of current time step
    std::vector<Feature>* GetFeatureVectorPtr(int index) { return &featureSequence[index]; }

private:
    vec3i predictRegion(int index, int direction, int mode); // Predict region t based on direction, returns offset
    void fillRegion(Feature& f, const vec3i& offset);        // Scanline algorithm - fills everything inside edge
    void expandRegion(Feature& f);                           // Grows edge where possible
    void shrinkRegion(Feature& f);                           // Shrinks edge where nescessary
    bool expandEdge(Feature& f, const vec3i& voxel);         // Sub-func inside expandRegion
    void shrinkEdge(Feature& f, const vec3i& voxel);         // Sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);                   // Update the feature vectors information after tracking
    void updateFeatureBoundary(Feature& f, const vec3i& voxel, int surface);
    Feature createNewFeature();

    float getOpacity(float value) { return tfMap[static_cast<int>(value * (tfRes-1))]; }

    float maskValue;  // Global mask value for newly detected features
    int tfRes;              // Default transfer function resolution
    int volumeSize;
    int timeLeft2Forward;
    int timeLeft2Backward;
    int numVoxelInFeature;

    vec3i blockDim;

    std::array<vec3i, 6> boundaryCtr;  // centroid of the boundary surface
    std::array<vec3i, 6> boundaryMin;  // min values of the boundary surface
    std::array<vec3i, 6> boundaryMax;  // max values of the boundary surface

    std::vector<float> data;        // Raw volume intensity value
    std::vector<float> mask;        // Mask volume, same size with a time step data
    std::vector<float> maskPrev;    // Mask volume, same size with a time step data
    std::vector<float> tfMap;       // Tranfer function setting
    std::vector<Feature> currentFeatures; // Features info in current time step
    std::vector<Feature> backup1Features; // ... in the 1st backup time step
    std::vector<Feature> backup2Features; // ... in the 2nd backup time step
    std::vector<Feature> backup3Features; // ... in the 3rd backup time step

    std::unordered_map<int, std::vector<Feature> > featureSequence;
};

#endif // FEATURETRACKER_H
