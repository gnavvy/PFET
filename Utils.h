#ifndef CONSTS_H
#define CONSTS_H

#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <map>

const float OPACITY_THRESHOLD  = 0.1;
const int MIN_NUM_VOXEL_IN_FEATURE = 10;
const int FT_DIRECT = 0;
const int FT_LINEAR = 1;
const int FT_POLYNO = 2;
const int FT_FORWARD  = 0;
const int FT_BACKWARD = 1;
const int DEFAULT_TF_RES = 1024;

using namespace std;

namespace util {
    template<class T>
    class vector3 {
    public:
        T x, y, z;
        vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        T*       GetPointer()                           { return &x; }
        T        VolumeSize()                           { return x * y * z; }
        float    MagnituteSquared()                     { return x*x + y*y + z*z; }
        float    Magnitute()                            { return sqrt((*this).MagnituteSquared()); }
        float    DistanceFrom(vector3 const& rhs) const { return (*this - rhs).Magnitute(); }
        vector3  operator -  ()                         { return vector3(-x, -y, -z); }
        vector3  operator +  (vector3 const& rhs) const { vector3 t(*this); t+=rhs; return t; }
        vector3  operator -  (vector3 const& rhs) const { vector3 t(*this); t-=rhs; return t; }
        vector3  operator *  (vector3 const& rhs) const { vector3 t(*this); t*=rhs; return t; }
        vector3  operator /  (vector3 const& rhs) const { vector3 t(*this); t/=rhs; return t; }
        vector3  operator *  (float scale)        const { vector3 t(*this); t*=scale; return t; }
        vector3  operator /  (float scale)        const { vector3 t(*this); t/=scale; return t; }
        vector3& operator += (vector3 const& rhs)       { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
        vector3& operator -= (vector3 const& rhs)       { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
        vector3& operator *= (vector3 const& rhs)       { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
        vector3& operator /= (vector3 const& rhs)       { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
        vector3& operator *= (float scale)              { x*=scale, y*=scale, z*=scale; return *this; }
        vector3& operator /= (float scale)              { x/=scale, y/=scale, z/=scale; return *this; }
        bool     operator == (vector3 const& rhs) const { return x==rhs.x && y==rhs.y && z==rhs.z; }
        bool     operator != (vector3 const& rhs) const { return !(*this == rhs); }
    };

    static inline string ltrim(const string &s) {    // trim string from left
        int start = s.find_first_not_of(' ');
        return s.substr(start, s.size() - start);
    }

    static inline string rtrim(const string &s) {    // trim string from right
        return s.substr(0, s.find_last_not_of(' ')+1);
    }

    static inline string trim(const string &s) {     // trim all whitesapces
        return ltrim(rtrim(s));
    }

    static inline bool ascending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
        return lhs.second < rhs.second;
    }

    static inline bool descending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
        return !ascending(lhs, rhs);
    }

    static inline int round(float f) {
        return static_cast<int>(floor(f + 0.5f));
    }
}

typedef util::vector3<int> vector3i;
typedef util::vector3<float> vector3f;

struct Feature {
    int             id;         // Unique ID for each feature
    float           maskValue;  // Used to record the color of the feature
    list<vector3i>  edgeVoxels; // Edge information of the feature
    list<vector3i>  bodyVoxels; // All the voxels in the feature
    vector3i        centroid;   // Centers position of the feature
};

struct Cluster {
    vector3i center;
    int numVoxels;
};

typedef unordered_map<int, float*> DataSequence;
typedef unordered_map<int, vector<Feature> > FeatureVectorSequence;

#endif // CONSTS_H
