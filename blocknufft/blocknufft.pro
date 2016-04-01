#-------------------------------------------------
#
# Project created by QtCreator 2015-09-09T20:48:54
#
#-------------------------------------------------

QT       -= core
QT       -= gui

TARGET = blocknufft
OBJECTS_DIR = build
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH += core
DEPENDPATH += core
VPATH += core

SOURCES += main.cpp \
    blocknufft3d.cpp \
    besseli.cpp \
    block3dspreader.cpp \
    blocknufft3d_c.cpp

QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp -std=c++11
LIBS += -fopenmp -lfftw3 -lfftw3_threads

HEADERS += \
    blocknufft3d.h \
    besseli.h \
    block3dspreader.h \
    blocknufft3d_c.h

HEADERS += qute.h
SOURCES += qute.cpp

#INCLUDEPATH += ../pebble/mdaio
#DEPENDPATH += ../pebble/mdaio
#VPATH += ../pebble/mdaio
#HEADERS += mda.h mdaio.h usagetracking.h
#SOURCES += mda.cpp mdaio.cpp usagetracking.cpp
