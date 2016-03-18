# -----------------------------------------------------------------------------
# DIAGErosionDilation project profile
#
# \file
# \author  Luc Nies
# \date    2016-03-04
# -----------------------------------------------------------------------------


TEMPLATE   = lib

TARGET     = DIAGErosionDilation

DESTDIR    = ../../../lib
DLLDESTDIR = ../../../lib

# Set high warn level (warn 4 on MSVC)
WARN = HIGH

# Add used projects here (see included pri files below for available projects)
CONFIG += dll ML

MLAB_PACKAGES += DIAG_General \
                 MeVisLab_Standard

# make sure that this file is included after CONFIG and MLAB_PACKAGES
include ($(MLAB_MeVis_Foundation)/Configuration/IncludePackages.pri)

DEFINES += DIAGEROSIONDILATION_EXPORTS

# Enable ML deprecated API warnings. To completely disable the deprecated API, change WARN to DISABLE.
DEFINES += ML_WARN_DEPRECATED

HEADERS += \
    DIAGErosionDilationInit.h \
    DIAGErosionDilationSystem.h \
    mlErosionDilation.h \

SOURCES += \
    DIAGErosionDilationInit.cpp \
    mlErosionDilation.cpp \