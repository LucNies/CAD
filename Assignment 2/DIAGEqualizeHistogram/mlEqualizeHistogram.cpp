//----------------------------------------------------------------------------------
//! The ML module class EqualizeHistogram.
/*!
// \file   
// \author  diag
// \date    2016-02-15
//
// 
*/
//----------------------------------------------------------------------------------

// Local includes
#include "mlEqualizeHistogram.h"
#include <iostream>
using namespace std;

ML_START_NAMESPACE

//! Implements code for the runtime type system of the ML
ML_MODULE_CLASS_SOURCE(EqualizeHistogram, Module);

//----------------------------------------------------------------------------------

EqualizeHistogram::EqualizeHistogram() : Module(1, 1)
{
  // Suppress calls of handleNotification on field changes to
  // avoid side effects during initialization phase.
  handleNotificationOff();

  // Reactivate calls of handleNotification on field changes.
  handleNotificationOn();


  // Activate inplace data buffers for output outputIndex and input inputIndex.
  // setOutputImageInplace(outputIndex, inputIndex);

  // Activate page data bypass from input inputIndex to output outputIndex.
  // Note that the module must still be able to calculate the output image.
  // setBypass(outputIndex, inputIndex);

}

//----------------------------------------------------------------------------------

void EqualizeHistogram::handleNotification(Field* field)
{
  // Handle changes of module parameters and input image fields here.
  bool touchOutputs = false;
  if (isInputImageField(field))
  {
    touchOutputs = true;
  }

  if (touchOutputs) 
  {
    // Touch all output image fields to notify connected modules.
    touchOutputImageFields();
  }
}

//----------------------------------------------------------------------------------

void EqualizeHistogram::calculateOutputImageProperties(int /*outputIndex*/, PagedImage* outputImage)
{
  // Change properties of output image outputImage here whose
  // defaults are inherited from the input image 0 (if there is one).
}

//----------------------------------------------------------------------------------

SubImageBox EqualizeHistogram::calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex)
{
  // Return region of input image inputIndex needed to compute region
  // outSubImgBox of output image outputIndex.
  return outputSubImageBox;
}


//----------------------------------------------------------------------------------

ML_CALCULATEOUTPUTSUBIMAGE_NUM_INPUTS_1_CPP(EqualizeHistogram);

template <typename T>
void EqualizeHistogram::calculateOutputSubImage(TSubImage<T>* outputSubImage, int outputIndex
                                     , TSubImage<T>* inputSubImage0
                                     )
{
  // Compute sub-image of output image outputIndex from input sub-images.

  // Clamp box of output image against image extent to avoid that unused areas are processed.
  const SubImageBox validOutBox = outputSubImage->getValidRegion();

  // Process all voxels of the valid region of the output page.
  ImageVector p;
  int min = 99999999;
  int max = 0;
  for (p.u=validOutBox.v1.u;  p.u<=validOutBox.v2.u;  ++p.u) {
    for (p.t=validOutBox.v1.t;  p.t<=validOutBox.v2.t;  ++p.t) {
      for (p.c=validOutBox.v1.c;  p.c<=validOutBox.v2.c;  ++p.c) {
        for (p.z=validOutBox.v1.z;  p.z<=validOutBox.v2.z;  ++p.z) {
          for (p.y=validOutBox.v1.y;  p.y<=validOutBox.v2.y;  ++p.y) {

            p.x = validOutBox.v1.x;
            // Get pointers to row starts of input and output sub-images.
            const T* inVoxel0 = inputSubImage0->getImagePointer(p);
            const MLint rowEnd   = validOutBox.v2.x;

            // Process all row voxels.
            for (; p.x <= rowEnd;  ++p.x, ++inVoxel0)
            {
			  min = (min > int(*inVoxel0)) ? int(*inVoxel0) : min;
			  max = (max < int(*inVoxel0)) ? int(*inVoxel0) : max;
            }
			
          }
        }
      }
    }
  }
  int range = max - min;
  mlDebug(range);
  std::vector<int> hist(range, 0);
  mlDebug(range);
  for (p.u=validOutBox.v1.u;  p.u<=validOutBox.v2.u;  ++p.u) {
    for (p.t=validOutBox.v1.t;  p.t<=validOutBox.v2.t;  ++p.t) {
      for (p.c=validOutBox.v1.c;  p.c<=validOutBox.v2.c;  ++p.c) {
        for (p.z=validOutBox.v1.z;  p.z<=validOutBox.v2.z;  ++p.z) {
          for (p.y=validOutBox.v1.y;  p.y<=validOutBox.v2.y;  ++p.y) {

            p.x = validOutBox.v1.x;
            // Get pointers to row starts of input and output sub-images.
            const T* inVoxel0 = inputSubImage0->getImagePointer(p);
			const MLint rowEnd   = validOutBox.v2.x;
			for (; p.x <= rowEnd;  ++p.x, ++inVoxel0)
			{
				hist[int(*inVoxel0) - min]++;
			}
		  }
		}
	  }
	}
  }
  
  mlDebug(range);
	std::vector<int> cum(range, 0);
  mlDebug(range);
	int sum = 0;
	for (int i=0; i <= int(hist.size()); i++)
	{
		sum += hist[i];
		cum[i]+=sum;
	} // sum now equals the total number of pixels
	cout << cum[0];
	std::vector<int> trans(range, 0);
	float constant = float(range)/float(sum);
	for (int i=0; i<= int(cum.size()); i++)
	{
		trans[i] = float(cum[i]) * constant;
	}
	
  mlDebug(range);
  for (p.u=validOutBox.v1.u;  p.u<=validOutBox.v2.u;  ++p.u) {
    for (p.t=validOutBox.v1.t;  p.t<=validOutBox.v2.t;  ++p.t) {
      for (p.c=validOutBox.v1.c;  p.c<=validOutBox.v2.c;  ++p.c) {
        for (p.z=validOutBox.v1.z;  p.z<=validOutBox.v2.z;  ++p.z) {
          for (p.y=validOutBox.v1.y;  p.y<=validOutBox.v2.y;  ++p.y) {
			  const MLint rowEnd   = validOutBox.v2.x;

            p.x = validOutBox.v1.x;
            // Get pointers to row starts of input and output sub-images.
            const T* inVoxel0 = inputSubImage0->getImagePointer(p);
            T*  outVoxel = outputSubImage->getImagePointer(p);
			for (; p.x <= rowEnd;  ++p.x, ++outVoxel, ++inVoxel0)
            {
			  *outVoxel = trans[*inVoxel0 - min];
			}
		  }
		}
	  }
	}
  }
  mlDebug(range);

}

ML_END_NAMESPACE