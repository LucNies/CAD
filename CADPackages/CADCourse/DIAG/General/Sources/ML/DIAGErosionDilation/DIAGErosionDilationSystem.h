//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  Luc Nies
// \date    2016-03-04
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGErosionDilationSystem_H
#define __DIAGErosionDilationSystem_H


// DLL export macro definition.
#ifdef DIAGEROSIONDILATION_EXPORTS
  // Use the DIAGEROSIONDILATION_EXPORT macro to export classes and functions.
  #define DIAGEROSIONDILATION_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define DIAGEROSIONDILATION_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif


#endif // __DIAGErosionDilationSystem_H
