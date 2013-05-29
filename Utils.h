#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <list>

const float OPACITY_THRESHOLD  = 0.2;
const int MIN_NUM_VOXEL_IN_FEATURE = 100;
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
        T*       GetPointer()                           { return &x; }
        T        Product()                              { return x * y * z; }
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
        bool     operator == (Vector3 const& rhs) const { return x==rhs.x && y==rhs.y && z==rhs.z; }
        bool     operator != (Vector3 const& rhs) const { return !(*this == rhs); }
    };
}

typedef util::Vector3<int> Vector3i;

struct Metadata {
    int      start;
    int      end;
    string   prefix;
    string   surfix;
    string   path;
    string   tfPath;
    string   timeFormat;
    Vector3i volumeDim;
};

struct Feature {
    int             id;             // Unique ID for each feature
    float           maskValue;      // Used to record the color of the feature
    list<Vector3i>  edgeVoxels;     // Edge information of the feature
    list<Vector3i>  bodyVoxels;     // All the voxels in the feature
    Vector3i        centroid;       // Centers position of the feature
};

typedef hash_map<int, float> IndexValueMap;
typedef hash_map<int, float*> DataSequence;
typedef hash_map<int, vector<Feature> > FeatureVectorSequence;

#endif // CONSTS_H
