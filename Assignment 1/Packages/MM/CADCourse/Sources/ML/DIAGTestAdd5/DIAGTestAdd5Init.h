//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-05
*/
//----------------------------------------------------------------------------------


#ifndef __DIAGTestAdd5Init_H
#define __DIAGTestAdd5Init_H


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int DIAGTestAdd5Init();

ML_END_NAMESPACE

#endif // __DIAGTestAdd5Init_H
