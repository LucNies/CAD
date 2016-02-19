//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-07
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGImageThresholdInit_H
#define __DIAGImageThresholdInit_H


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int DIAGImageThresholdInit();

ML_END_NAMESPACE

#endif // __DIAGImageThresholdInit_H
