//----------------------------------------------------------------------------------
//! The ML module class ToCSV.
/*!
// \file    
// \author  Luc Nies
// \date    2016-02-22
//
// Ass3
*/
//----------------------------------------------------------------------------------


#ifndef __mlToCSV_H
#define __mlToCSV_H


// Local includes
#include "DIAGToCSVSystem.h"

// ML includes
#include <mlModuleIncludes.h>

ML_START_NAMESPACE


//! Ass3
class DIAGTOCSV_EXPORT ToCSV : public Module
{
public:

  //! Constructor.
  ToCSV();

  //! Handles field changes of the field \p field.
  virtual void handleNotification (Field* field);

private:

  // ----------------------------------------------------------
  //! \name Module field declarations
  //@{
  // ----------------------------------------------------------

  BoolField* _Positive_NegativeFld;
  BoolField* _Train_TestFld;
  BoolField* _applyFld;

  //average
  FloatField* _input0Fld;
  //Contrast
  FloatField* _input1Fld;
  //Coarsness
  FloatField* _input2Fld;
  //Corralation
  FloatField* _input3Fld;
  //
  FloatField* _input4Fld;
  //! 
  IntField* _input5Fld;
  //@}

  // Implements interface for the runtime type system of the ML.
  ML_MODULE_CLASS_HEADER(ToCSV)

};


ML_END_NAMESPACE

#endif // __mlToCSV_H