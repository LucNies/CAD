# -----------------------------------------------------------------------------
# DIAGEqualizeHistogram project profile
#
# \file
# \author  diag
# \date    2016-02-15
# -----------------------------------------------------------------------------


TEMPLATE   = lib

TARGET     = DIAGEqualizeHistogram

DESTDIR    = ../../../lib
DLLDESTDIR = ../../../lib

# Set high warn level (warn 4 on MSVC)
WARN = HIGH

# Add used projects here (see included pri files below for available projects)
CONFIG += dll ML

MLAB_PACKAGES += DIAG_ApplicationBase \
                 MeVisLab_Standard

# make sure that this file is included after CONFIG and MLAB_PACKAGES
include ($(MLAB_MeVis_Foundation)/Configuration/IncludePackages.pri)

DEFINES += DIAGEQUALIZEHISTOGRAM_EXPORTS

# Enable ML deprecated API warnings. To completely disable the deprecated API, change WARN to DISABLE.
DEFINES += ML_WARN_DEPRECATED

HEADERS += \
    DIAGEqualizeHistogramInit.h \
    DIAGEqualizeHistogramSystem.h \
    mlEqualizeHistogram.h \

SOURCES += \
    DIAGEqualizeHistogramInit.cpp \
    mlEqualizeHistogram.cpp \