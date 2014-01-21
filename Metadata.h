#ifndef METADATA_H
#define METADATA_H

#include "Utils.h"

class Metadata {
public:
    int      start()      const { return start_; }
    int      end()        const { return end_; }
    string   prefix()     const { return prefix_; }
    string   suffix()     const { return suffix_; }
    string   path()       const { return path_; }
    string   tfPath()     const { return tfPath_; }
    string   timeFormat() const { return timeFormat_; }
    vector3i volumeDim()  const { return volumeDim_; }

    Metadata(const string &fpath);
   ~Metadata();

private:
    int      start_;
    int      end_;
    string   prefix_;
    string   suffix_;
    string   path_;
    string   tfPath_;
    string   timeFormat_;
    vector3i volumeDim_;
};

#endif // METADATA_H
