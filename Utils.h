#ifndef CONSTS_H
#define CONSTS_H

#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <list>

const float DIST_THRESHOLD = 4.0;
const float OPACITY_THRESHOLD  = 0.1;

const int MIN_NUM_VOXEL_IN_FEATURE = 10;
const int FT_DIRECT = 0;
const int FT_LINEAR = 1;
const int FT_POLYNO = 2;
const int FT_FORWARD  = 0;
const int FT_BACKWARD = 1;
const int DEFAULT_TF_RES = 1024;

// Surfaces
const int SURFACE_NULL   = -1;  // default
const int LEFT   = 0;   // x = 0
const int RIGHT  = 1;   // x = xs-1
const int BOTTOM = 2;   // y = 0
const int TOP    = 3;   // y = ys-1
const int FRONT  = 4;   // z = 0
const int BACK   = 5;   // z = zs-1

using namespace std;

namespace util {
    template<class T>
    class vec3 {
    public:
        T x, y, z;
        vec3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        T*    data()                              { return &x; }
        T     volumeSize()                        { return x * y * z; }
        float magnituteSquared()                  { return x*x + y*y + z*z; }
        float magnitute()                         { return sqrt((*this).magnituteSquared()); }
        float distanceFrom(vec3 const& rhs) const { return (*this - rhs).magnitute(); }
        vec3  operator -  ()                      { return vec3(-x, -y, -z); }
        vec3  operator +  (vec3 const& rhs) const { vec3 t(*this); t+=rhs; return t; }
        vec3  operator -  (vec3 const& rhs) const { vec3 t(*this); t-=rhs; return t; }
        vec3  operator *  (vec3 const& rhs) const { vec3 t(*this); t*=rhs; return t; }
        vec3  operator /  (vec3 const& rhs) const { vec3 t(*this); t/=rhs; return t; }
        vec3  operator *  (float scale)     const { vec3 t(*this); t*=scale; return t; }
        vec3  operator /  (float scale)     const { vec3 t(*this); t/=scale; return t; }
        vec3& operator += (vec3 const& rhs)       { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
        vec3& operator -= (vec3 const& rhs)       { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
        vec3& operator *= (vec3 const& rhs)       { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
        vec3& operator /= (vec3 const& rhs)       { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
        vec3& operator *= (float scale)           { x*=scale, y*=scale, z*=scale; return *this; }
        vec3& operator /= (float scale)           { x/=scale, y/=scale, z/=scale; return *this; }
        bool  operator == (vec3 const& rhs) const { return x==rhs.x && y==rhs.y && z==rhs.z; }
        bool  operator != (vec3 const& rhs) const { return !(*this == rhs); }
    };

    static inline string ltrim(const string &s) {    // trim string from left
        int start = s.find_first_not_of(' ');
        return s.substr(start, s.size() - start);
    }

    static inline string rtrim(const string &s) {    // trim string from right
        return s.substr(0, s.find_last_not_of(' ')+1);
    }

    static inline string trim(const string &s) {     // trim all white spaces
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

    static inline vec3<int> min(const vec3<int>& v1, const vec3<int>& v2) {
        return vec3<int>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
    }

    static inline vec3<int> max(const vec3<int>& v1, const vec3<int>& v2) {
        return vec3<int>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
    }
}

typedef util::vec3<int> vec3i;

class Edge {
public:
    int id, start, end;
    vec3i centroid;

    bool operator ==(Edge const& rhs) const {
        Edge lhs(*this);
        if (lhs.id==rhs.id && lhs.start==rhs.start && lhs.end==rhs.end) {
            return true;
        } else return false;
    }
};  // start --id--> end @ centroid

struct Feature {
    int   id;        // Unique ID for each feature
    int   maskValue; // color id of the feature
    vec3i ctr;       // Centroid position of the feature
    vec3i min;       // Minimum position (x, y, z) of the feature
    vec3i max;       // Maximum position (x, y, z) of the feature
    std::array<vec3i, 6> boundaryCtr; // center point on each boundary surface
    std::array<vec3i, 6> boundaryMin; // min value on each boundary surface
    std::array<vec3i, 6> boundaryMax; // max value on each boundary surface
    std::array<int, 6> numVoxelOnSurface; // number of voxels on each boundary surface
    std::list<vec3i> edgeVoxels; // Edge information of the feature
    std::list<vec3i> bodyVoxels; // All the voxels in the feature
    std::vector<int> touchedSurfaces;
};

#endif // CONSTS_H
