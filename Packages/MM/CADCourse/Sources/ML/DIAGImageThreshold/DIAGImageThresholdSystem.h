//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-07
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGImageThresholdSystem_H
#define __DIAGImageThresholdSystem_H


// DLL export macro definition.
#ifdef DIAGIMAGETHRESHOLD_EXPORTS
  // Use the DIAGIMAGETHRESHOLD_EXPORT macro to export classes and functions.
  #define DIAGIMAGETHRESHOLD_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define DIAGIMAGETHRESHOLD_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif


#endif // __DIAGImageThresholdSystem_H
