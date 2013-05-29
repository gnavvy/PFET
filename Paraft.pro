QMAKE_CC     = gcc
QMAKE_CXX    = g++
#QMAKE_CC     = /usr/local/bin/mpicc
#QMAKE_CXX    = /usr/local/bin/mpic++
#LIBS        += -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lutil
INCLUDEPATH += /usr/local/include \
               /usr/llvm-gcc-4.2/lib/gcc/i686-apple-darwin11/4.2.1/include

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp
LIBS += -L/usr/local/lib \
        -L/usr/llvm-gcc-4.2/lib/gcc/i686-apple-darwin11/4.2.1/x86_64 -lgomp

SOURCES += \
    Main.cpp \
    DataManager.cpp \
    FeatureTracker.cpp \
    BlockController.cpp

HEADERS += \
    DataManager.h \
    FeatureTracker.h \
    BlockController.h \
    Utils.h
