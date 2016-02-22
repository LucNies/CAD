//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-22
*/
//----------------------------------------------------------------------------------


// Local includes
#include "DIAGToCSVSystem.h"

// Include definition of ML_INIT_LIBRARY.
#include "mlLibraryInitMacros.h"

// Include all module headers ...
#include "mlToCSV.h"


ML_START_NAMESPACE

//----------------------------------------------------------------------------------
//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
//----------------------------------------------------------------------------------
int DIAGToCSVInit()
{
  ToCSV::initClass();
  // Add initClass calls from all other modules here.

  return 1;
}

ML_END_NAMESPACE


//! Calls the init method implemented above during load of shared library.
ML_INIT_LIBRARY(DIAGToCSVInit)