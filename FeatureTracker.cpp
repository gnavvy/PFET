#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(vec3i dim) : blockDim(dim), maskValue(0), tfRes(1024) {
    volumeSize = blockDim.volumeSize();
    mask = std::vector<float>(volumeSize);
    maskPrev = std::vector<float>(volumeSize);
}

FeatureTracker::~FeatureTracker() {}

void FeatureTracker::ExtractAllFeatures() {
    for (int z = 0; z < blockDim.z; ++z) {
        for (int y = 0; y < blockDim.y; ++y) {
            for (int x = 0; x < blockDim.x; ++x) {
                int index = GetVoxelIndex(vec3i(x, y, z));
                if (mask[index] > 0) { 
                    continue; // point already within a feature
                }
                int tfindex = (int)(data[index] * (tfRes-1));
                if (tfMap[tfindex] >= OPACITY_THRESHOLD) {
                    FindNewFeature(vec3i(x,y,z));
                }
            }
        }
    }
}

Feature FeatureTracker::createNewFeature() {
    Feature f;
    f.id        = -1;
    f.maskValue = -1;
    f.ctr        = vec3i();
    f.min        = blockDim;
    f.max        = vec3i();
    for (int surface = 0; surface < 6; ++surface) {
        f.boundaryCtr[surface] = vec3i();
        f.boundaryMin[surface] = vec3i();
        f.boundaryMax[surface] = vec3i();
        f.numVoxelOnSurface[surface] = 0;
    }
    f.edgeVoxels.clear();
    f.bodyVoxels.clear();
    f.touchedSurfaces.clear();
    return f;
}

void FeatureTracker::FindNewFeature(vec3i seed) {
    maskValue++;

    Feature f = createNewFeature();
    f.maskValue = maskValue;
    f.edgeVoxels.push_back(seed);

    expandRegion(f);

    if (static_cast<int>(f.bodyVoxels.size()) < MIN_NUM_VOXEL_IN_FEATURE) {
        maskValue--; 
        return;
    }

    currentFeatures.push_back(f);
    backup1Features = currentFeatures;
    backup2Features = currentFeatures;
    backup3Features = currentFeatures;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    if (tfMap.size() == 0 || tfRes <= 0) {
        std::cout << "Set TF pointer first." << std::endl;
        exit(EXIT_FAILURE);
    }

    data.assign(pData, pData+volumeSize);

    // backup mask to maskPrev then clear mask
    maskPrev.clear();
    maskPrev.swap(mask);

    for (auto i = 0; i < currentFeatures.size(); ++i) {
        Feature f = currentFeatures[i];

        vec3i offset = predictRegion(i, direction, mode);
        fillRegion(f, offset);
        shrinkRegion(f);
        expandRegion(f);

        if (static_cast<int>(f.bodyVoxels.size()) < MIN_NUM_VOXEL_IN_FEATURE) {
            // erase feature from list when it becomes too small
            currentFeatures.erase(currentFeatures.begin() + i);
            continue;
        } else {
            currentFeatures[i] = f;    
        }
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline vec3i FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timeLeft2Backward : timeLeft2Forward;

    vec3i offset;
    Feature b1f = backup1Features[index];
    Feature b2f = backup2Features[index];
    Feature b3f = backup3Features[index];

    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    offset = b2f.ctr - b1f.ctr;
                } else {  // Tracking forward as default
                    offset = b3f.ctr - b2f.ctr;
                }
                for (auto& voxel : b3f.edgeVoxels) {
                    voxel += offset;
                    voxel = util::min(voxel, blockDim-vec3i(1,1,1));   // x, y, z at most dim-1
                    voxel = util::max(voxel, vec3i());  // x, y, z at least 0
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    offset = b3f.ctr*2 - b2f.ctr*3 + b1f.ctr;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        offset = b2f.ctr - b1f.ctr;
                    } else {  // Tracking forward as default
                        offset = b3f.ctr - b2f.ctr;
                    }
                }
                for (auto& voxel : b3f.edgeVoxels) {
                    voxel += offset;
                    voxel = util::min(voxel, blockDim-vec3i(1,1,1));   // x, y, z at most dim-1
                    voxel = util::max(voxel, vec3i());  // x, y, z at least 0
                }
            }
        break;
    }
    return offset;
}

inline void FeatureTracker::fillRegion(Feature &f, const vec3i& offset) {
    // predicted to be on edge
    for (auto const &voxel : f.edgeVoxels) {
        int index = GetVoxelIndex(voxel);
        if (mask[index] == 0.0) {
            mask[index] = static_cast<float>(f.maskValue);
        }
        f.bodyVoxels.push_back(voxel);
        f.ctr += voxel;
    }

    // currently not on edge but previously on edge
    for (auto const &voxel : f.edgeVoxels) {
        vec3i voxelPrev = voxel - offset;
        int index = GetVoxelIndex(voxel);
        int indexPrev = GetVoxelIndex(voxelPrev);
        if (voxel.x >= 0 && voxel.x <= blockDim.x && voxelPrev.x >= 0 && voxelPrev.x <= blockDim.x &&
            voxel.y >= 0 && voxel.y <= blockDim.y && voxelPrev.y >= 0 && voxelPrev.y <= blockDim.y &&
            voxel.z >= 0 && voxel.z <= blockDim.z && voxelPrev.z >= 0 && voxelPrev.z <= blockDim.z &&
            mask[index] == 0.0 && maskPrev[indexPrev] == static_cast<float>(f.maskValue)) {

            // mark voxels that: 1. currently = 1; or 2. currently = 0 but previously = 1;
            mask[index] = static_cast<float>(f.maskValue);
            f.bodyVoxels.push_back(voxel);
            f.ctr += voxel;
        }
    }
}

inline void FeatureTracker::shrinkRegion(Feature &f) {
    // mark all edge points as 0
    while (!f.edgeVoxels.empty()) {
        vec3i voxel = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        shrinkEdge(f, voxel);
    }

    while (!f.bodyVoxels.empty()) {
        vec3i voxel = f.bodyVoxels.front();
        f.bodyVoxels.pop_front();

        int index = GetVoxelIndex(voxel);
        bool voxelOnEdge = false;
        if (getOpacity(data[index]) < OPACITY_THRESHOLD) {
            voxelOnEdge = false;
            // if point is invisible, mark its adjacent points as 0
            shrinkEdge(f, voxel);                                               // center
            if (++voxel.x < blockDim.x) { shrinkEdge(f, voxel); } voxel.x--;   // right
            if (++voxel.y < blockDim.y) { shrinkEdge(f, voxel); } voxel.y--;   // top
            if (++voxel.z < blockDim.z) { shrinkEdge(f, voxel); } voxel.z--;   // back
            if (--voxel.x >= 0)          { shrinkEdge(f, voxel); } voxel.x++;   // left
            if (--voxel.y >= 0)          { shrinkEdge(f, voxel); } voxel.y++;   // bottom
            if (--voxel.z >= 0)          { shrinkEdge(f, voxel); } voxel.z++;   // front
        } else if (mask[index] == 0.0f) { voxelOnEdge = true; }

        if (voxelOnEdge) { 
            f.edgeVoxels.push_back(voxel); 
        }
    }

    for (auto const &voxel : f.edgeVoxels) {
        int index = GetVoxelIndex(voxel);
        if (mask[index] != static_cast<float>(f.maskValue)) {
            mask[index] = static_cast<float>(f.maskValue);
            f.bodyVoxels.push_back(voxel);
            f.ctr += voxel;
        }
    }
}

inline void FeatureTracker::shrinkEdge(Feature& f, const vec3i& voxel) {
    int index = GetVoxelIndex(voxel);
    if (mask[index] == static_cast<float>(f.maskValue)) {
        mask[index] = 0.0;  // shrink
        auto it = std::find(f.bodyVoxels.begin(), f.bodyVoxels.end(), voxel);
        if (it != f.bodyVoxels.end()) {
            f.bodyVoxels.erase(it);    
            f.edgeVoxels.push_back(voxel);
            f.ctr -= voxel;
        }
    }
}

inline void FeatureTracker::expandRegion(Feature& f) {
    std::list<vec3i> tempVoxels;    // to store updated edge voxels
    while (!f.edgeVoxels.empty()) {
        vec3i voxel = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        bool voxelOnEdge = false;
        if (++voxel.x < blockDim.x) { voxelOnEdge |= expandEdge(f, voxel); } voxel.x--;  // right
        if (++voxel.y < blockDim.y) { voxelOnEdge |= expandEdge(f, voxel); } voxel.y--;  // top
        if (++voxel.z < blockDim.z) { voxelOnEdge |= expandEdge(f, voxel); } voxel.z--;  // front
        if (--voxel.x >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.x++;  // left
        if (--voxel.y >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.y++;  // bottom
        if (--voxel.z >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.z++;  // back

        // if any one of the six neighboring points is not on edge, the original
        // seed voxel is still considered as on edge and will be put back to edge list
        if (voxelOnEdge) { 
            tempVoxels.push_back(voxel); 
        }
    }
    f.edgeVoxels.swap(tempVoxels);

    // update feature info - accumulative
    if (f.bodyVoxels.size() != 0) {
        f.ctr /= f.bodyVoxels.size();
        f.id = GetVoxelIndex(f.ctr);
    }

    for (int surface = 0; surface < 6; ++surface) {
        if (f.numVoxelOnSurface[surface] != 0) {
            f.boundaryCtr[surface] /= f.numVoxelOnSurface[surface];
            if (std::find(f.touchedSurfaces.begin(), f.touchedSurfaces.end(), surface) == 
                f.touchedSurfaces.end()) {
                f.touchedSurfaces.push_back(surface);
            }           
        }
    }
}

inline bool FeatureTracker::expandEdge(Feature& f, const vec3i& voxel) {
    int index = GetVoxelIndex(voxel);

    if (mask[index] > 0 || getOpacity(data[index]) < OPACITY_THRESHOLD) {
        // this neighbor voxel is already labeled, or the opacity is not large enough
        // to be labeled as within the feature, so the original seed is still on edge.
        return true;
    }

    // update feature info
    mask[index] = static_cast<float>(f.maskValue);
    f.min = util::min(f.min, voxel);
    f.max = util::max(f.max, voxel);
    f.ctr += voxel;  // averaged later
    if (voxel.x == 0) { updateFeatureBoundary(f, voxel, LEFT);   }
    if (voxel.y == 0) { updateFeatureBoundary(f, voxel, BOTTOM); }
    if (voxel.z == 0) { updateFeatureBoundary(f, voxel, FRONT);  }
    if (voxel.x == blockDim.x-1) { updateFeatureBoundary(f, voxel, RIGHT); }
    if (voxel.y == blockDim.y-1) { updateFeatureBoundary(f, voxel, TOP);   }
    if (voxel.z == blockDim.z-1) { updateFeatureBoundary(f, voxel, BACK);  }
    f.edgeVoxels.push_back(voxel);
    f.bodyVoxels.push_back(voxel);
    // f.touchedSurfaces is updated after the expandRegion() is done

    // the original seed voxel is no longer on edge for this neighboring direction
    return false;
}

void FeatureTracker::updateFeatureBoundary(Feature& f, const vec3i& voxel, int surface) {
    f.boundaryMin[surface] = util::min(f.boundaryMin[surface], voxel);
    f.boundaryMax[surface] = util::max(f.boundaryMax[surface], voxel);
    f.boundaryCtr[surface] += voxel;
    f.numVoxelOnSurface[surface]++;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1Features = backup2Features;
    backup2Features = backup3Features;
    backup3Features = currentFeatures;

    if (direction == FT_FORWARD) {
        if (timeLeft2Forward  < 3) ++timeLeft2Forward;
        if (timeLeft2Backward > 0) --timeLeft2Backward;
    } else {    // direction is either FORWARD or BACKWARD
        if (timeLeft2Forward  > 0) --timeLeft2Forward;
        if (timeLeft2Backward < 3) ++timeLeft2Backward;
    }
}
