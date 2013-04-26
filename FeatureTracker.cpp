#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(Vector3i dim) : blockDim(dim) {
    maskValue = 0.0f;
    tfRes = 0;
    pTfMap = NULL;
    volumeSize = blockDim.Product();

    pMaskCurrent = new float[volumeSize]();
    pMaskPrevious = new float[volumeSize]();

    if (pMaskCurrent == NULL || pMaskPrevious == NULL) {
        return;
    }
}

FeatureTracker::~FeatureTracker() {
    delete [] pMaskCurrent;
    delete [] pMaskPrevious;

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();

    if (featureSequence.size() > 0) {
        for (FeatureVectorSequence::iterator it = featureSequence.begin(); it != featureSequence.end(); it++) {
            vector<Feature> featureVector = it->second;
            for (size_t i = 0; i < featureVector.size(); i++) {
                featureVector.at(i).SurfacePoints.clear();
                featureVector.at(i).InnerPoints.clear();
            }
        }
    }
}

void FeatureTracker::Reset() {
    maskValue = 0.0f;
    numVoxelinFeature = 0;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
    sumCoordinateValue.x = sumCoordinateValue.y = sumCoordinateValue.z = 0;

    std::fill(pMaskCurrent, pMaskCurrent+volumeSize, 0);
    std::fill(pMaskPrevious, pMaskPrevious+volumeSize, 0);

    currentFeaturesHolder.clear();
    backup1FeaturesHolder.clear();
    backup2FeaturesHolder.clear();
    backup3FeaturesHolder.clear();

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();
}

void FeatureTracker::ExtractAllFeatures() {
    for (int z = 0; z < blockDim.z; z++) {
        for (int y = 0; y < blockDim.y; y++) {
            for (int x = 0; x < blockDim.x; x++) {
                int index = GetVoxelIndex(Vector3i(x, y, z));
                if (pMaskCurrent[index] != 0)   // point already within a feature
                    continue;                   // most points should stop here
                int tfIndex = (int)(pVolumeData[index] * (float)(tfRes-1));
                if (pTfMap[tfIndex] >= LOW_THRESHOLD && pTfMap[tfIndex] <= HIGH_THRESHOLD) {
                    FindNewFeature(Vector3i(x,y,z));
                }
            }
        }
    }
}

void FeatureTracker::FindNewFeature(Vector3i point) {
    sumCoordinateValue = point;

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();

    /////////////////////////////////
    // Only one point now, take as edge point
    numVoxelinFeature = 1;
    surfacePoints.push_back(point);
    /////////////////////////////////

    int index = GetVoxelIndex(point);
    if (pMaskCurrent[index] == 0) {
        maskValue += 1.0f;
        pMaskCurrent[index] = maskValue;
    } else {
        return;
    }

    /////////////////////////////////
    expandRegion(maskValue);
    /////////////////////////////////

    if (innerPoints.size() < (size_t)FEATURE_MIN_VOXEL_NUM) {
        maskValue -= 1.0f;
        return;
    }

    Feature newFeature; {
        newFeature.ID               = GetVoxelIndex(centroid);
        newFeature.Centroid         = centroid;
        newFeature.SurfacePoints    = surfacePoints;
        newFeature.InnerPoints      = innerPoints;
        newFeature.MaskValue        = maskValue;
    }

    currentFeaturesHolder.push_back(newFeature);
    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    pVolumeData = pData;
    innerPoints.clear();

    if (pTfMap == NULL || tfRes <= 0) {
        cout << "Set TF pointer first." << endl; exit(3);
    }

    // save current 0-1 matrix to previous, then clear current maxtrix
    std::copy(pMaskCurrent, pMaskCurrent+volumeSize, pMaskPrevious);
    std::fill(pMaskCurrent, pMaskCurrent+volumeSize, 0);

    for (size_t i = 0; i < currentFeaturesHolder.size(); i++) {
        Feature f = currentFeaturesHolder[i];

        predictRegion(i, direction, mode);
        fillRegion(f.MaskValue);
        shrinkRegion(f.MaskValue);
        expandRegion(f.MaskValue);

        f.ID              = GetVoxelIndex(centroid);
        f.Centroid        = centroid;
        f.SurfacePoints   = surfacePoints;
        f.InnerPoints     = innerPoints;
        currentFeaturesHolder[i] = f;
        innerPoints.clear();
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline void FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timestepsAvailableBackward : timestepsAvailableForward;

    delta.x = delta.y = delta.z = 0;
    Feature b1f = backup1FeaturesHolder[index];
    Feature b2f = backup2FeaturesHolder[index];
    Feature b3f = backup3FeaturesHolder[index];

    int tmp;
    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    delta.x = b2f.Centroid.x - b1f.Centroid.x;
                    delta.y = b2f.Centroid.y - b1f.Centroid.y;
                    delta.z = b2f.Centroid.z - b1f.Centroid.z;
                } else {    // Tracking forward as default
                    delta.x = b3f.Centroid.x - b2f.Centroid.x;
                    delta.y = b3f.Centroid.y - b2f.Centroid.y;
                    delta.z = b3f.Centroid.z - b2f.Centroid.z;
                }

                for (list<Vector3i>::iterator p = b3f.SurfacePoints.begin(); p != b3f.SurfacePoints.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    delta.x = b3f.Centroid.x*2 - b2f.Centroid.x*3 + b1f.Centroid.x;
                    delta.y = b3f.Centroid.y*2 - b2f.Centroid.y*3 + b1f.Centroid.y;
                    delta.z = b3f.Centroid.z*2 - b2f.Centroid.z*3 + b1f.Centroid.z;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        delta.x = b2f.Centroid.x - b1f.Centroid.x;
                        delta.y = b2f.Centroid.y - b1f.Centroid.y;
                        delta.z = b2f.Centroid.z - b1f.Centroid.z;
                    } else {    // Tracking forward as default
                        delta.x = b3f.Centroid.x - b2f.Centroid.x;
                        delta.y = b3f.Centroid.y - b2f.Centroid.y;
                        delta.z = b3f.Centroid.z - b2f.Centroid.z;
                    }
                }

                for (list<Vector3i>::iterator p = b3f.SurfacePoints.begin(); p != b3f.SurfacePoints.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
    }
}

inline void FeatureTracker::fillRegion(float maskValue) {
    sumCoordinateValue.x = 0;
    sumCoordinateValue.y = 0;
    sumCoordinateValue.z = 0;
    numVoxelinFeature = 0;
    int index = 0;
    int indexPrev = 0;

    list<Vector3i>::iterator p;

    // predicted to be on edge
    for (p = surfacePoints.begin(); p != surfacePoints.end(); p++) {
        index = GetVoxelIndex(*p);
        if (pMaskCurrent[index] == 0) {
            pMaskCurrent[index] = maskValue;
        }
        sumCoordinateValue += (*p);
        innerPoints.push_back(*p);
        numVoxelinFeature++;
    }

    // currently not on edge but previously on edge
    for (p = surfacePoints.begin(); p != surfacePoints.end(); p++) {
        index = GetVoxelIndex(*p);
        indexPrev = GetVoxelIndex(Vector3i((*p).x-delta.x, (*p).y-delta.y, (*p).z-delta.z));
        (*p).x++;
        while ((*p).x >= 0 && (*p).x <= blockDim.x && (*p).x - delta.x >= 0 && (*p).x - delta.x <= blockDim.x &&
               (*p).y >= 0 && (*p).y <= blockDim.y && (*p).y - delta.y >= 0 && (*p).y - delta.y <= blockDim.y &&
               (*p).z >= 0 && (*p).z <= blockDim.z && (*p).z - delta.z >= 0 && (*p).z - delta.z <= blockDim.z &&
               pMaskCurrent[index] == 0 && pMaskPrevious[indexPrev] == maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            pMaskCurrent[index] = maskValue;
            sumCoordinateValue += (*p);
            innerPoints.push_back(*p);
            numVoxelinFeature++;
            (*p).x++;
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline void FeatureTracker::shrinkRegion(float maskValue) {
    Vector3i point;
    int index = 0;
    int indexCurr = 0;
    bool isPointOnEdge;

    // mark all edge points as 0
    while (surfacePoints.empty() == false) {
        point = surfacePoints.front();
        shrinkEdge(point, maskValue);
        surfacePoints.pop_front();
    }

    while (dataPointList.empty() == false) {
        point = dataPointList.front();
        dataPointList.pop_front();

        index = GetVoxelIndex(point);
        if (getOpacity(pVolumeData[index]) < lowerThreshold || getOpacity(pVolumeData[index]) > upperThreshold) {
            isPointOnEdge = false;
            // if point is invisible, mark its adjacent points as 0                    //
            shrinkEdge(point, maskValue);                                              // center
            if (++point.x < blockDim.x) { shrinkEdge(point, maskValue); } point.x--;   // right
            if (++point.y < blockDim.y) { shrinkEdge(point, maskValue); } point.y--;   // top
            if (++point.z < blockDim.z) { shrinkEdge(point, maskValue); } point.z--;   // back
            if (--point.x > 0)          { shrinkEdge(point, maskValue); } point.x++;   // left
            if (--point.y > 0)          { shrinkEdge(point, maskValue); } point.y++;   // bottom
            if (--point.z > 0)          { shrinkEdge(point, maskValue); } point.z++;   // front
        } else if (pMaskCurrent[index] == 0) { isPointOnEdge = true; }

        if (isPointOnEdge == true) { surfacePoints.push_back(point); }
    }

    for (list<Vector3i>::iterator p = surfacePoints.begin(); p != surfacePoints.end(); ++p) {
        index = GetVoxelIndex((*p));
        indexCurr = GetVoxelIndex(point);
        if (pMaskCurrent[indexCurr] != maskValue) {
            sumCoordinateValue += (*p);
            innerPoints.push_back(*p);
            numVoxelinFeature++;
            pMaskCurrent[indexCurr] = maskValue;
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline void FeatureTracker::shrinkEdge(Vector3i point, float maskValue) {
    int index = GetVoxelIndex(point);
    if (pMaskCurrent[index] == maskValue) {
        pMaskCurrent[index] = 0;    // shrink
        sumCoordinateValue -= point;
        numVoxelinFeature--;
        for (list<Vector3i>::iterator p = innerPoints.begin(); p != innerPoints.end(); p++) {
            if (point.x == (*p).x && point.y == (*p).y && point.z == (*p).z) {
                innerPoints.erase(p); break;
            }
        }
        dataPointList.push_back(point);
    }
}

// Grow edge where possible for all the features in the CurrentFeaturesHolder
// Say if we have several features in one time step, the number we call GrowRegion is the number of features
// Each time before we call this function, we should copy the edges points of one feature we want to grow in edge
inline void FeatureTracker::expandRegion(float maskValue) {
    // put edge points to feature body
    while (surfacePoints.empty() == false) {
        dataPointList.push_back(surfacePoints.front());
        surfacePoints.pop_front();
    } // edgePointList should be empty

    while (dataPointList.empty() == false) {
        Vector3i point = dataPointList.front();
        dataPointList.pop_front();
        bool onBoundary = false;
        if (++point.x < blockDim.x) { onBoundary |= expandEdge(point, maskValue); } point.x--;  // right
        if (++point.y < blockDim.y) { onBoundary |= expandEdge(point, maskValue); } point.y--;  // top
        if (++point.z < blockDim.z) { onBoundary |= expandEdge(point, maskValue); } point.z--;  // front
        if (--point.x > 0)          { onBoundary |= expandEdge(point, maskValue); } point.x++;  // left
        if (--point.y > 0)          { onBoundary |= expandEdge(point, maskValue); } point.y++;  // bottom
        if (--point.z > 0)          { onBoundary |= expandEdge(point, maskValue); } point.z++;  // back
        if (onBoundary)         { surfacePoints.push_back(point); }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline bool FeatureTracker::expandEdge(Vector3i point, float maskValue) {
    int index = GetVoxelIndex(point);
    if (pMaskCurrent[index] != 0) {
        return false;   // not on edge, inside feature, no need to adjust
    }

    if (getOpacity(pVolumeData[index]) >= lowerThreshold && getOpacity(pVolumeData[index]) <= upperThreshold) {
        pMaskCurrent[index] = maskValue;
        dataPointList.push_back(point);
        innerPoints.push_back(point);
        sumCoordinateValue += point;
        numVoxelinFeature++;
        return false;
    } else {
        return true;
    }
}

void FeatureTracker::SetCurrentFeatureInfo(vector<Feature> *pFeatures) {
    currentFeaturesHolder.clear();
    for (size_t i = 0; i < pFeatures->size(); i++) {
        currentFeaturesHolder.push_back(pFeatures->at(i));
    }

    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1FeaturesHolder = backup2FeaturesHolder;
    backup2FeaturesHolder = backup3FeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    if (direction == FT_FORWARD) {
        if (timestepsAvailableForward  < 3) timestepsAvailableForward++;
        if (timestepsAvailableBackward > 0) timestepsAvailableBackward--;
    } else {    // direction is either FORWARD or BACKWARD
        if (timestepsAvailableForward  > 0) timestepsAvailableForward--;
        if (timestepsAvailableBackward < 3) timestepsAvailableBackward++;
    }
}
