#include "Metadata.h"

Metadata::Metadata(const string &fpath) {
    ifstream meta(fpath.c_str());
    if (!meta) {
        cout << "cannot read meta file: " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    while (getline(meta, line)) {
        size_t pos = line.find('=');
        if (pos == line.npos) { continue; }

        string value = util::trim(line.substr(pos+1));
        if (line.find("start") != line.npos) {
            start_ = atoi(value.c_str());
        } else if (line.find("end") != line.npos) {
            end_ = atoi(value.c_str());
        } else {
            // remove leading & trailing chars () or ""
            value = value.substr(1, value.size()-2);

            if (line.find("prefix") != line.npos) {
                prefix_ = value;
            } else if (line.find("suffix") != line.npos) {
                suffix_ = value;
            } else if (line.find("path") != line.npos) {
                path_ = value;
            } else if (line.find("tfPath") != line.npos) {
                tfPath_ = value;
            } else if (line.find("timeFormat") != line.npos) {
                timeFormat_ = value;
            } else if (line.find("volumeDim") != line.npos) {
                vector<int> dim;
                size_t pos = 0;
                while ((pos = value.find(',')) != value.npos) {
                    dim.push_back(atoi(util::trim(value.substr(0, pos)).c_str()));
                    value.erase(0, pos+1);
                }
                dim.push_back(atoi(util::trim(value).c_str()));

                if (dim.size() != 3) {
                    cout << "incorrect volumeDim format" << endl;
                    exit(EXIT_FAILURE);
                }
                volumeDim_ = vector3i(dim[0], dim[1], dim[2]);
            }
        }
    }
}

Metadata::~Metadata() {}
