//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  diag
// \date    2016-02-15
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGEqualizeHistogramSystem_H
#define __DIAGEqualizeHistogramSystem_H


// DLL export macro definition.
#ifdef DIAGEQUALIZEHISTOGRAM_EXPORTS
  // Use the DIAGEQUALIZEHISTOGRAM_EXPORT macro to export classes and functions.
  #define DIAGEQUALIZEHISTOGRAM_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define DIAGEQUALIZEHISTOGRAM_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif


#endif // __DIAGEqualizeHistogramSystem_H
