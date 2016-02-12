# -----------------------------------------------------------------------------
# DIAGTestAdd5 project profile
#
# \file
# \author  Luc Nies
# \date    2016-02-05
# -----------------------------------------------------------------------------


TEMPLATE   = lib

TARGET     = DIAGTestAdd5

DESTDIR    = ../../../lib
DLLDESTDIR = ../../../lib

# Set high warn level (warn 4 on MSVC)
WARN = HIGH

# Add used projects here (see included pri files below for available projects)
CONFIG += dll ML

MLAB_PACKAGES += MM_CADCourse \
                 MeVisLab_Standard

# make sure that this file is included after CONFIG and MLAB_PACKAGES
include ($(MLAB_MeVis_Foundation)/Configuration/IncludePackages.pri)

DEFINES += DIAGTESTADD5_EXPORTS

# Enable ML deprecated API warnings. To completely disable the deprecated API, change WARN to DISABLE.
DEFINES += ML_WARN_DEPRECATED

HEADERS += \
    DIAGTestAdd5Init.h \
    DIAGTestAdd5System.h \
    mlTestAdd5.h \

SOURCES += \
    DIAGTestAdd5Init.cpp \
    mlTestAdd5.cpp \