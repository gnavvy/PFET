#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <list>

const int FEATURE_MIN_VOXEL_NUM = 10;

const float LOW_THRESHOLD  = 0.2;
const float HIGH_THRESHOLD = 1.0;

const int FT_DIRECT = 0;
const int FT_LINEAR = 1;
const int FT_POLYNO = 2;
const int FT_FORWARD  = 0;
const int FT_BACKWARD = 1;

using namespace std;

namespace util {
    template<class T>
    class Vector3 {
    public:
        T x, y, z;
        Vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        int*     GetPointer()                           { return &x; }
        int      Product()                              { return x * y * z; }
        float    Magnitute()                            { return sqrt(x*x + y*y + z*z); }
        float    DistanceFrom(Vector3 const& rhs) const { return (*this - rhs).Magnitute(); }
        Vector3  operator -  ()                         { return Vector3(-x, -y, -z); }
        Vector3  operator +  (Vector3 const& rhs) const { Vector3 t(*this); t+=rhs; return t; }
        Vector3  operator -  (Vector3 const& rhs) const { Vector3 t(*this); t-=rhs; return t; }
        Vector3  operator *  (Vector3 const& rhs) const { Vector3 t(*this); t*=rhs; return t; }
        Vector3  operator /  (Vector3 const& rhs) const { Vector3 t(*this); t/=rhs; return t; }
        Vector3  operator *  (int scale)          const { Vector3 t(*this); t*=scale; return t; }
        Vector3  operator /  (int scale)          const { Vector3 t(*this); t/=scale; return t; }
        Vector3& operator += (Vector3 const& rhs)       { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
        Vector3& operator -= (Vector3 const& rhs)       { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
        Vector3& operator *= (Vector3 const& rhs)       { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
        Vector3& operator /= (Vector3 const& rhs)       { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
        Vector3& operator *= (int scale)                { x*=scale, y*=scale, z*=scale; return *this; }
        Vector3& operator /= (int scale)                { x/=scale, y/=scale, z/=scale; return *this; }
    };
}

typedef util::Vector3<int> Vector3i;

struct Metadata {
    int             start;
    int             end;
    string          prefix;
    string          surfix;
    string          path;
    string          tfPath;
    Vector3i        volumeDim;
};

struct Feature {
    int             ID;             // Unique ID for each feature
    float           MaskValue;      // Used to record the color of the feature
    list<Vector3i>  SurfacePoints;  // Edge information of the feature
    list<Vector3i>  InnerPoints;    // All the voxels in the feature
    Vector3i        Centroid;       // Centers position of the feature
    Vector3i        Min;            // Minimum position (x,y,z) on boundary
    Vector3i        Max;            // Maximum position (x,y,z) on boundary
};

typedef hash_map<int, float> IndexValueMap;
typedef hash_map<int, float*> DataSequence;
typedef hash_map<int, vector<Feature> > FeatureVectorSequence;

#endif // CONSTS_H
