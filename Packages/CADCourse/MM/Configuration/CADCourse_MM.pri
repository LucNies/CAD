isEmpty(CADCourse_MM_PRI_INCLUDED) {
  message ( loading CADCourse_MM.pri )
}
# **InsertLicense** code
# -----------------------------------------------------------------------------
# CADCourse_MM prifile
#
# \file    CADCourse_MM.pri
# \author  Team666
# \date    2016-02-19
#
# 
#
# -----------------------------------------------------------------------------

# include guard against multiple inclusion
isEmpty(CADCourse_MM_PRI_INCLUDED) {

CADCourse_MM_PRI_INCLUDED = 1

# -- System -------------------------------------------------------------

include( $(MLAB_MeVis_Foundation)/Configuration/SystemInit.pri )

# -- Define local PACKAGE variables -------------------------------------

PACKAGE_ROOT    = $$(MLAB_CADCourse_MM)
PACKAGE_SOURCES = "$$(MLAB_CADCourse_MM)"/Sources

# Add package library path
LIBS          += -L"$${PACKAGE_ROOT}"/lib

# -- Projects -------------------------------------------------------------

# NOTE: Add projects below to make them available to other projects via the CONFIG mechanism

# You can use this example template for typical projects:
#MLMyProject {
#  CONFIG_FOUND += MLMyProject
#  INCLUDEPATH += $${PACKAGE_SOURCES}/ML/MLMyProject
#  win32:LIBS += MLMyProject$${d}.lib
#  unix:LIBS += -lMLMyProject$${d}
#}

# -- ML Projects -------------------------------------------------------------

# -- Inventor Projects -------------------------------------------------------

# -- Shared Projects ---------------------------------------------------------

# End of projects ------------------------------------------------------------

}