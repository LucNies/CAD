isEmpty(DIAG_General_PRI_INCLUDED) {
  message ( loading DIAG_General.pri )
}
# **InsertLicense** code
# -----------------------------------------------------------------------------
# DIAG_General prifile
#
# \file    DIAG_General.pri
# \author  Luc
# \date    2016-03-04
#
# 
#
# -----------------------------------------------------------------------------

# include guard against multiple inclusion
isEmpty(DIAG_General_PRI_INCLUDED) {

DIAG_General_PRI_INCLUDED = 1

# -- System -------------------------------------------------------------

include( $(MLAB_MeVis_Foundation)/Configuration/SystemInit.pri )

# -- Define local PACKAGE variables -------------------------------------

PACKAGE_ROOT    = $$(MLAB_DIAG_General)
PACKAGE_SOURCES = "$$(MLAB_DIAG_General)"/Sources

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