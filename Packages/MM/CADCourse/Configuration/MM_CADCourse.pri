isEmpty(MM_CADCourse_PRI_INCLUDED) {
  message ( loading MM_CADCourse.pri )
}
# **InsertLicense** code
# -----------------------------------------------------------------------------
# MM_CADCourse prifile
#
# \file    MM_CADCourse.pri
# \author  DIAG Student
# \date    2016-02-05
#
# Test
#
# -----------------------------------------------------------------------------

# include guard against multiple inclusion
isEmpty(MM_CADCourse_PRI_INCLUDED) {

MM_CADCourse_PRI_INCLUDED = 1

# -- System -------------------------------------------------------------

include( $(MLAB_MeVis_Foundation)/Configuration/SystemInit.pri )

# -- Define local PACKAGE variables -------------------------------------

PACKAGE_ROOT    = $$(MLAB_MM_CADCourse)
PACKAGE_SOURCES = "$$(MLAB_MM_CADCourse)"/Sources

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