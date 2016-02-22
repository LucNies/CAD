//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-22
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGToCSVInit_H
#define __DIAGToCSVInit_H


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int DIAGToCSVInit();

ML_END_NAMESPACE

#endif // __DIAGToCSVInit_H
