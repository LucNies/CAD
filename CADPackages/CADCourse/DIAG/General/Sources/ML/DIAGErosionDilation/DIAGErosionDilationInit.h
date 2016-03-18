//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Luc Nies
// \date    2016-03-04
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGErosionDilationInit_H
#define __DIAGErosionDilationInit_H


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int DIAGErosionDilationInit();

ML_END_NAMESPACE

#endif // __DIAGErosionDilationInit_H
