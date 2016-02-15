//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  diag
// \date    2016-02-15
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGEqualizeHistogramInit_H
#define __DIAGEqualizeHistogramInit_H


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int DIAGEqualizeHistogramInit();

ML_END_NAMESPACE

#endif // __DIAGEqualizeHistogramInit_H
