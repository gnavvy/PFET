#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Utils.h"

using namespace std;

class FeatureTracker {

public:
    FeatureTracker(Vector3i dim);
    ~FeatureTracker() ;

    void Reset();
    void ExtractAllFeatures();

    // Set seed at current time step. FindNewFeature will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
    void FindNewFeature(Vector3i point);

    // Track forward based on the center points of the features at the last time step
    void TrackFeature(float* pData, int direction, int mode);

    // Get mask matrix of current time step
    float* GetMaskPointer()  { return pMaskCurrent; }
    float* GetTFOpacityMap() { return pTfMap; }

    void SaveExtractedFeatures(int index) { featureSequence[index] = currentFeaturesHolder; }

    // accessor
    void SetDataPointer(float* pData) { pVolumeData = pData; }
    void SetTFOpacityMap(float* map) { pTfMap = map; }
    void SetTFResolution(int res) { tfRes = res; }
    void SetThresholds(float lower, float upper) { lowerThreshold = lower; upperThreshold = upper; }
    int GetVoxelIndex(const Vector3i &voxel) { return blockDim.x*blockDim.y*voxel.z+blockDim.x*voxel.y+voxel.x; }
    int GetTFResolution() { return tfRes; }

    // Get all features information of current time step
    vector<Feature>* GetFeatureVectorPointer(int index) { return &featureSequence[index]; }
    void SetCurrentFeatureInfo(vector<Feature>* pFeature);

private:
    void predictRegion(int index, int direction, int mode); // predict region t based on direction
    void fillRegion(float maskValue);                       // scanline algorithm - fills everything inside edge
    void expandRegion(float maskValue);                     // grows edge where possible
    void shrinkRegion(float maskValue);                     // shrinks edge where nescessary
    bool expandEdge(Vector3i point, float maskValue);      // sub-func inside expandRegion
    void shrinkEdge(Vector3i point, float maskValue);      // sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);                  // Update the feature vectors information after tracking

    float getOpacity(float value) { return pTfMap[(int)(value * (tfRes-1))]; }

    float* pMaskCurrent;        // Mask volume, same size with a time step data
    float* pMaskPrevious;       // Mask volume, for backward time step when tracking forward & backward
    float* pVolumeData;         // Raw volume intensity value
    float* pTfMap;
    float lowerThreshold;
    float upperThreshold;
    float maskValue;

    int tfRes;
    int volumeSize;
    int numVoxelinFeature;
    int timestepsAvailableForward;
    int timestepsAvailableBackward;

    // FeatureInfo
    list<Vector3i> dataPointList;  // Active queue
    list<Vector3i> surfacePoints;  // For edge points saving
    list<Vector3i> innerPoints;    // Hold the temp buffer of the voxel position

    Vector3i blockDim;
    Vector3i centroid;              // center point of a single feature
    Vector3i sumCoordinateValue;    // Sum of the voxel values of the feature
    Vector3i delta;

    vector<Feature> currentFeaturesHolder; // Features info in current time step
    vector<Feature> backup1FeaturesHolder; // ... in the 1st backup time step
    vector<Feature> backup2FeaturesHolder; // ... in the 2nd backup time step
    vector<Feature> backup3FeaturesHolder; // ... in the 3rd backup time step

    FeatureVectorSequence featureSequence;
};

#endif // FEATURETRACKER_H
