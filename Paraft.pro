QMAKE_CXX       =  g++-4.8
QMAKE_CXXFLAGS  = -std=c++11
INCLUDEPATH     = -I/usr/local/include
LIBS            = -L/usr/local/lib -lm

QMAKE_LINK       = $$QMAKE_CXX

SOURCES += \
    Main.cpp \
    DataManager.cpp \
    FeatureTracker.cpp \
    BlockController.cpp \
    Metadata.cpp

HEADERS += \
    DataManager.h \
    FeatureTracker.h \
    BlockController.h \
    Utils.h \
    Metadata.h

OTHER_FILES += \
    vorts.config \
    jet.config \
    supervoxel.config
