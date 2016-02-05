//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-05
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGTestAdd5System_H
#define __DIAGTestAdd5System_H


// DLL export macro definition.
#ifdef DIAGTESTADD5_EXPORTS
  // Use the DIAGTESTADD5_EXPORT macro to export classes and functions.
  #define DIAGTESTADD5_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define DIAGTESTADD5_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif


#endif // __DIAGTestAdd5System_H
