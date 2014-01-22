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
const int SURFACE_LEFT   = 0;   // x = 0
const int SURFACE_RIGHT  = 1;   // x = xs
const int SURFACE_BOTTOM = 2;   // y = 0
const int SURFACE_TOP    = 3;   // y = ys
const int SURFACE_FRONT  = 4;   // z = 0
const int SURFACE_BACK   = 5;   // z = zs

using namespace std;

namespace util {
    template<class T>
    class vector3 {
    public:
        T x, y, z;
        vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        T*       data()                                 { return &x; }
        T        volumeSize()                           { return x * y * z; }
        float    magnituteSquared()                     { return x*x + y*y + z*z; }
        float    magnitute()                            { return sqrt((*this).magnituteSquared()); }
        float    distanceFrom(vector3 const& rhs) const { return (*this - rhs).magnitute(); }
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

typedef util::vector3<int> Vec3i;
typedef util::vector3<float> Vec3f;

class Edge {
public:
    int id, start, end;
    Vec3i centroid;

    bool operator ==(Edge const& rhs) const {
        Edge lhs(*this);
        if (lhs.id==rhs.id && lhs.start==rhs.start && lhs.end==rhs.end) {
            return true;
        } else return false;
    }
};  // start --id--> end @ centroid

struct Feature {
    int   id;                   // Unique ID for each feature
    float maskValue;            // Used to record the color of the feature
    Vec3i centroid;             // Centers position of the feature
    Vec3i min;                  // Minimum position (x,y,z) on boundary
    Vec3i max;                  // Maximum position (x,y,z) on boundary
    Vec3i boundaryCentroid[6];  // center point on boundary surface
    Vec3i boundaryMin[6];       // min value on boundary surface
    Vec3i boundaryMax[6];       // max value on boundary surface
    std::list<Vec3i> edgeVoxels; // Edge information of the feature
    std::list<Vec3i> bodyVoxels; // All the voxels in the feature
    std::vector<int> touchedSurfaces;
};

#endif // CONSTS_H
