#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Utils.h"

using namespace std;

class FeatureTracker {

public:
    FeatureTracker(Vector3i dim);
    ~FeatureTracker() ;

    void ExtractAllFeatures();

    // Set seed at current time step. FindNewFeature will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
    void FindNewFeature(Vector3i seed);

    // Track forward based on the center points of the features at the last time step
    void TrackFeature(float* pData, int direction, int mode);
    void SaveExtractedFeatures(int index)           { featureSequence[index] = currentFeatures; }
    void SetDataPointer(float* pData)               { pVolumeData = pData; }
    void SetTFMap(float* map)                       { pTfMap = map; }
    void SetTFResolution(int res)                   { tfRes = res; }
    float* GetMaskPointer()                         { return pMask; }
    float* GetTFOpacityMap()                        { return pTfMap; }
    int GetTFResolution()                           { return tfRes; }
    int GetVoxelIndex(const Vector3i &voxel)        { return blockDim.x*blockDim.y*voxel.z+blockDim.x*voxel.y+voxel.x; }

    // Get all features information of current time step
    vector<Feature>* GetFeatureVectorPointer(int index) { return &featureSequence[index]; }

private:
    Vector3i predictRegion(int index, int direction, int mode); // predict region t based on direction, returns offset
    void fillRegion(Feature &f, const Vector3i &offset);    // scanline algorithm - fills everything inside edge
    void expandRegion(Feature &f);  // grows edge where possible
    void shrinkRegion(Feature &f);  // shrinks edge where nescessary
    bool expandEdge(Feature &f, const Vector3i &seed); // sub-func inside expandRegion
    void shrinkEdge(Feature &f, const Vector3i &seed); // sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);              // Update the feature vectors information after tracking

    float getOpacity(float value) { return pTfMap[(int)(value * (tfRes-1))]; }

    float* pMask;               // Mask volume, same size with a time step data
    float* pMaskPrevious;       // Mask volume, for backward time step when tracking forward & backward
    float* pVolumeData;         // Raw volume intensity value
    float* pTfMap;              // Tranfer function setting
    float  globalMaskValue;     // Global mask value for newly detected features

    int tfRes;
    int volumeSize;
    int timeLeft2Forward;
    int timeLeft2Backward;

    Vector3i blockDim;

    vector<Feature> currentFeatures; // Features info in current time step
    vector<Feature> backup1Features; // ... in the 1st backup time step
    vector<Feature> backup2Features; // ... in the 2nd backup time step
    vector<Feature> backup3Features; // ... in the 3rd backup time step

    FeatureVectorSequence featureSequence;
};

#endif // FEATURETRACKER_H
