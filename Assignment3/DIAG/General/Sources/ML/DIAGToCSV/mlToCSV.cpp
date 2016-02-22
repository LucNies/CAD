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

// Local includes
#include "mlToCSV.h"
#include <iostream>
#include <fstream>
using namespace std;

ML_START_NAMESPACE

//! Implements code for the runtime type system of the ML
ML_MODULE_CLASS_SOURCE(ToCSV, Module);

//----------------------------------------------------------------------------------

ToCSV::ToCSV() : Module(0, 0)
{
  // Suppress calls of handleNotification on field changes to
  // avoid side effects during initialization phase.
  handleNotificationOff();

  // Add fields to the module and set their values.
  _input0Fld = addFloat("input0", 0);
  _input1Fld = addFloat("input1", 0);
  _input2Fld = addFloat("input2", 0);
  _input3Fld = addFloat("input3", 0);
  _input4Fld = addFloat("input4", 0);
  _input5Fld = addInt("input5", 0);


  _Positive_NegativeFld = addBool("Positive_Negative", false);
  _Train_TestFld = addBool("Train_Test", false);
  _applyFld = addBool("apply", false);

  // Reactivate calls of handleNotification on field changes.
  handleNotificationOn();


  // Activate inplace data buffers for output outputIndex and input inputIndex.
  // setOutputImageInplace(outputIndex, inputIndex);

  // Activate page data bypass from input inputIndex to output outputIndex.
  // Note that the module must still be able to calculate the output image.
  // setBypass(outputIndex, inputIndex);

}

//----------------------------------------------------------------------------------

void ToCSV::handleNotification(Field* field)
{
	string pathName = "C:/Users/Luc/Documents/GitHub/CAD/";
	string fileName = pathName + "featuredata.csv";
  // Handle changes of module parameters and input image fields here.
	if (_applyFld == field)
	{
		ofstream file;
		ifstream outFile(fileName);
		if(!outFile)//file does not exist yet
		{
			file.open(fileName);
			file << "Average" << "," << "Contrast" << "," << "Coarsness" << "," << "Correlation" << "," << "Variation" << "," << "k"<<  endl;
			file << (*_input0Fld).getFloatValue() << "," << _input1Fld << "," << _input2Fld << "," << _input3Fld << "," << _input4Fld << "," << _input5Fld << endl;
		}
		else
		{
			file.open(fileName);
			file << (*_input0Fld).getFloatValue() << "," << (*_input1Fld).getFloatValue() << "," << (*_input2Fld).getFloatValue() << "," << (*_input3Fld).getFloatValue() << "," << (*_input4Fld).getFloatValue() << "," << (*_input5Fld).getIntValue() << endl;
		}

		file.close();

	}
}

ML_END_NAMESPACE