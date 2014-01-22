#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(Vec3i dim) : blockDim_(dim) {
    globalMaskValue_ = 0.0f;
    tfRes_ = 1024;
    volumeSize_ = blockDim_.volumeSize();
    mask_ = std::vector<float>(volumeSize_);
    maskPrev_ = std::vector<float>(volumeSize_);
}

FeatureTracker::~FeatureTracker() {
    if (featureSequence_.size() > 0) {
        for (auto it = featureSequence_.begin(); it != featureSequence_.end(); ++it) {
            vector<Feature> featureVector = it->second;
            for (size_t i = 0; i < featureVector.size(); ++i) {
                Feature f = featureVector[i];
                f.edgeVoxels.clear();
                f.bodyVoxels.clear();
            }
        }
    }
}

void FeatureTracker::ExtractAllFeatures() {
    for (int z = 0; z < blockDim_.z; ++z) {
        for (int y = 0; y < blockDim_.y; ++y) {
            for (int x = 0; x < blockDim_.x; ++x) {
                int index = GetVoxelIndex(Vec3i(x, y, z));
                if (mask_[index] > 0) continue; // point already within a feature
                int tfIndex = (int)(data_[index] * (tfRes_-1));
                if (tfMap_[tfIndex] >= OPACITY_THRESHOLD) {
                    FindNewFeature(Vec3i(x,y,z));
                }
            }
        }
    }
}

void FeatureTracker::FindNewFeature(Vec3i seed) {
    Feature f; {
        f.id         = 0;
        f.centroid   = Vec3i();
        f.edgeVoxels = list<Vec3i>();
        f.bodyVoxels = list<Vec3i>();
        f.maskValue  = globalMaskValue_ += 1.0f;
    }

    f.edgeVoxels.push_back(seed);
    expandRegion(f);

    if (f.bodyVoxels.size() < (size_t)MIN_NUM_VOXEL_IN_FEATURE) {
        globalMaskValue_ -= 1.0f; return;
    }

    currentFeatures_.push_back(f);
    backup1Features_ = currentFeatures_;
    backup2Features_ = currentFeatures_;
    backup3Features_ = currentFeatures_;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    if (tfMap_.size() == 0 || tfRes_ <= 0) {
        std::cout << "Set TF pointer first." << std::endl;
        exit(EXIT_FAILURE);
    }

    data_.assign(pData, pData+volumeSize_);

    // save current 0-1 matrix to previous, then clear current maxtrix
    maskPrev_ = mask_;
    std::fill(mask_.begin(), mask_.end(), 0.0f);

    for (size_t i = 0; i < currentFeatures_.size(); ++i) {
        Feature f = currentFeatures_[i];

        Vec3i offset = predictRegion(i, direction, mode);
        fillRegion(f, offset);
        shrinkRegion(f);
        expandRegion(f);

        if (f.bodyVoxels.empty()) {
            // todo f.numVoxels < MIN_NUM_VOXEL_IN_FEATURE
            // currentFeaturesHolder.erase(currentFeaturesHolder.begin()+i);
            continue;
        }

        f.centroid /= f.bodyVoxels.size();
        f.id = GetVoxelIndex(f.centroid);
        currentFeatures_[i] = f;
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline Vec3i FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timeLeft2Backward_ : timeLeft2Forward_;

    Vec3i off;   // offset
    Feature b1f = backup1Features_[index];
    Feature b2f = backup2Features_[index];
    Feature b3f = backup3Features_[index];

    int tmp;
    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    off = b2f.centroid - b1f.centroid;
                } else {  // Tracking forward as default
                    off = b3f.centroid - b2f.centroid;
                }
                for (list<Vec3i>::iterator p = b3f.edgeVoxels.begin(); p != b3f.edgeVoxels.end(); ++p) {
                    tmp = (*p).x + (int)floor(off.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim_.x ? tmp : blockDim_.x-1);
                    tmp = (*p).y + (int)floor(off.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim_.y ? tmp : blockDim_.y-1);
                    tmp = (*p).z + (int)floor(off.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim_.z ? tmp : blockDim_.z-1);
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    off = b3f.centroid*2 - b2f.centroid*3 + b1f.centroid;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        off = b2f.centroid - b1f.centroid;
                    } else {  // Tracking forward as default
                        off = b3f.centroid - b2f.centroid;
                    }
                }
                for (list<Vec3i>::iterator p = b3f.edgeVoxels.begin(); p != b3f.edgeVoxels.end(); ++p) {
                    tmp = (*p).x + (int)floor(off.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim_.x ? tmp : blockDim_.x-1);
                    tmp = (*p).y + (int)floor(off.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim_.y ? tmp : blockDim_.y-1);
                    tmp = (*p).z + (int)floor(off.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim_.z ? tmp : blockDim_.z-1);
                }
            }
        break;
    }
    return off;
}

inline void FeatureTracker::fillRegion(Feature &f, const Vec3i &offset) {
    // predicted to be on edge
    for (list<Vec3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); ++p) {
        int index = GetVoxelIndex(*p);
        if (mask_[index] == 0) {
            mask_[index] = f.maskValue;
        }
        f.bodyVoxels.push_back(*p);
        f.centroid += (*p);
    }

    // currently not on edge but previously on edge
    for (list<Vec3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); ++p) {
        int index = GetVoxelIndex(*p);
        int indexPrev = GetVoxelIndex((*p)-offset);
        while ((*p).x >= 0 && (*p).x <= blockDim_.x && (*p).x - offset.x >= 0 && (*p).x - offset.x <= blockDim_.x &&
               (*p).y >= 0 && (*p).y <= blockDim_.y && (*p).y - offset.y >= 0 && (*p).y - offset.y <= blockDim_.y &&
               (*p).z >= 0 && (*p).z <= blockDim_.z && (*p).z - offset.z >= 0 && (*p).z - offset.z <= blockDim_.z &&
               mask_[index] == 0 && maskPrev_[indexPrev] == f.maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            mask_[index] = f.maskValue;
            f.bodyVoxels.push_back(*p);
            f.centroid += (*p);
        }
    }
}

inline void FeatureTracker::shrinkRegion(Feature &f) {
    // mark all edge points as 0
    while (f.edgeVoxels.empty() == false) {
        Vec3i seed = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        shrinkEdge(f, seed);
    }

    while (!f.bodyVoxels.empty()) {
        Vec3i seed = f.bodyVoxels.front();
        f.bodyVoxels.pop_front();

        int index = GetVoxelIndex(seed);
        bool seedOnEdge = false;
        if (getOpacity(data_[index]) < OPACITY_THRESHOLD) {
            seedOnEdge = false;
            // if point is invisible, mark its adjacent points as 0
            shrinkEdge(f, seed);                                            // center
            if (++seed.x < blockDim_.x) { shrinkEdge(f, seed); } seed.x--;   // right
            if (++seed.y < blockDim_.y) { shrinkEdge(f, seed); } seed.y--;   // top
            if (++seed.z < blockDim_.z) { shrinkEdge(f, seed); } seed.z--;   // back
            if (--seed.x >= 0)          { shrinkEdge(f, seed); } seed.x++;   // left
            if (--seed.y >= 0)          { shrinkEdge(f, seed); } seed.y++;   // bottom
            if (--seed.z >= 0)          { shrinkEdge(f, seed); } seed.z++;   // front
        } else if (mask_[index] == 0.0f) { seedOnEdge = true; }

        if (seedOnEdge) { f.edgeVoxels.push_back(seed); }
    }

    for (list<Vec3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); ++p) {
        int index = GetVoxelIndex(*p);
        if (mask_[index] != f.maskValue) {
            mask_[index] = f.maskValue;
            f.bodyVoxels.push_back(*p);
            f.centroid += (*p);
        }
    }
}

inline void FeatureTracker::shrinkEdge(Feature &f, const Vec3i &seed) {
    int index = GetVoxelIndex(seed);
    if (mask_[index] == f.maskValue) {
        mask_[index] = 0;  // shrink
        list<Vec3i>::iterator p = find(f.bodyVoxels.begin(), f.bodyVoxels.end(), seed);
        f.bodyVoxels.erase(p);
        f.edgeVoxels.push_back(seed);
        f.centroid -= seed;
    }
}

inline void FeatureTracker::expandRegion(Feature &f) {
    list<Vec3i> tempVoxels;    // to store updated edge voxels
    while (!f.edgeVoxels.empty()) {
        Vec3i seed = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        bool seedOnEdge = false;
        if (++seed.x < blockDim_.x) { seedOnEdge |= expandEdge(f, seed); } seed.x--;  // right
        if (++seed.y < blockDim_.y) { seedOnEdge |= expandEdge(f, seed); } seed.y--;  // top
        if (++seed.z < blockDim_.z) { seedOnEdge |= expandEdge(f, seed); } seed.z--;  // front
        if (--seed.x >= 0)          { seedOnEdge |= expandEdge(f, seed); } seed.x++;  // left
        if (--seed.y >= 0)          { seedOnEdge |= expandEdge(f, seed); } seed.y++;  // bottom
        if (--seed.z >= 0)          { seedOnEdge |= expandEdge(f, seed); } seed.z++;  // back

        // if any one of the six neighboring points is not on edge, the original
        // seed point is still considered as on edge and will be put back to edge list
        if (seedOnEdge)            { tempVoxels.push_back(seed); }
    }

    f.edgeVoxels.swap(tempVoxels);
}

inline bool FeatureTracker::expandEdge(Feature &f, const Vec3i &seed) {
    int index = GetVoxelIndex(seed);

    // this neighbor voxel is already labeled, or the opacity is not large enough to
    // to be labeled as within the feature, so the original seed is still on edge.
    if (mask_[index] > 0 || getOpacity(data_[index]) < OPACITY_THRESHOLD) {
        return true;
    }

    mask_[index] = f.maskValue;
    f.edgeVoxels.push_back(seed);
    f.bodyVoxels.push_back(seed);
    f.centroid += seed;

    // the original seed is no longer on edge for this neighboring direction
    return false;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1Features_ = backup2Features_;
    backup2Features_ = backup3Features_;
    backup3Features_ = currentFeatures_;

    if (direction == FT_FORWARD) {
        if (timeLeft2Forward_  < 3) ++timeLeft2Forward_;
        if (timeLeft2Backward_ > 0) --timeLeft2Backward_;
    } else {    // direction is either FORWARD or BACKWARD
        if (timeLeft2Forward_  > 0) --timeLeft2Forward_;
        if (timeLeft2Backward_ < 3) ++timeLeft2Backward_;
    }
}
