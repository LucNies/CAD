//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-22
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGToCSVSystem_H
#define __DIAGToCSVSystem_H


// DLL export macro definition.
#ifdef DIAGTOCSV_EXPORTS
  // Use the DIAGTOCSV_EXPORT macro to export classes and functions.
  #define DIAGTOCSV_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define DIAGTOCSV_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif


#endif // __DIAGToCSVSystem_H
