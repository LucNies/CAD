//----------------------------------------------------------------------------------
//! The ML module class ErosionDilation.
/*!
// \file   
// \author  Luc Nies
// \date    2016-03-04
//
// 
*/
//----------------------------------------------------------------------------------

// Local includes
#include "mlErosionDilation.h"
#include <math.h> 

using namespace std;
ML_START_NAMESPACE

//! Implements code for the runtime type system of the ML
ML_MODULE_CLASS_SOURCE(ErosionDilation, Module);

//----------------------------------------------------------------------------------
int radius;
bool erosion;

ErosionDilation::ErosionDilation() : Module(1, 1)
{
  // Suppress calls of handleNotification on field changes to
  // avoid side effects during initialization phase.
  handleNotificationOff();

  // Add fields to the module and set their values.
  _radiusFld = addInt("radius", 1);
  _erosionFld = addBool("erosion", false);

  // Reactivate calls of handleNotification on field changes.
  handleNotificationOn();


  // Activate inplace data buffers for output outputIndex and input inputIndex.
  // setOutputImageInplace(outputIndex, inputIndex);

  // Activate page data bypass from input inputIndex to output outputIndex.
  // Note that the module must still be able to calculate the output image.
  // setBypass(outputIndex, inputIndex);

}

//----------------------------------------------------------------------------------

void ErosionDilation::handleNotification(Field* field)
{
  // Handle changes of module parameters and input image fields here.
  bool touchOutputs = false;
  if (isInputImageField(field))
  {
    touchOutputs = true;
  }
  else if (field == _radiusFld)
  {
	radius = (*_radiusFld).getIntValue();
    touchOutputs = true;
  }
  else if (field == _erosionFld)
  {
	erosion = (*_erosionFld).getBoolValue();
    touchOutputs = true;
  }

  if (touchOutputs) 
  {
    // Touch all output image fields to notify connected modules.
    touchOutputImageFields();
  }
}

//----------------------------------------------------------------------------------

void ErosionDilation::calculateOutputImageProperties(int /*outputIndex*/, PagedImage* outputImage)
{
  // Change properties of output image outputImage here whose
  // defaults are inherited from the input image 0 (if there is one).

  // Set output image to a fixed type.
  //outputImage->setDataType(MLdoubleType);

  // Specify input sub-image data types (otherwise the above output data type is used for all input sub-images).
  //outputImage->setInputSubImageDataType(0, getInputImage(0)->getDataType());
}

//----------------------------------------------------------------------------------

SubImageBox ErosionDilation::calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex)
{
  // Return region of input image inputIndex needed to compute region
  // outSubImgBox of output image outputIndex.
  return outputSubImageBox;
}


//----------------------------------------------------------------------------------

ML_CALCULATEOUTPUTSUBIMAGE_NUM_INPUTS_1_CPP(ErosionDilation);

template <typename T>
void ErosionDilation::calculateOutputSubImage(TSubImage<T>* outputSubImage, int outputIndex
                                     , TSubImage<T>* inputSubImage0
                                     )
{

	ImageVector p;
	const SubImageBox validOutBox = outputSubImage->getValidRegion();
  // Compute sub-image of output image outputIndex from input sub-images.
	for (p.u = validOutBox.v1.u; p.u <= validOutBox.v2.u; ++p.u) {
		for (p.t = validOutBox.v1.t; p.t <= validOutBox.v2.t; ++p.t) {
			for (p.c = validOutBox.v1.c; p.c <= validOutBox.v2.c; ++p.c) {
				for (p.z = validOutBox.v1.z; p.z <= validOutBox.v2.z; ++p.z) {
					for (p.y = validOutBox.v1.y; p.y <= validOutBox.v2.y; ++p.y) {
						T*  outVoxel = outputSubImage->getImagePointer(p);
						*outVoxel = 0;
					}
				}
			}
		}
	}


  // Clamp box of output image against image extent to avoid that unused areas are processed.
  
  mlDebug("radius: " << radius);
  // Process all voxels of the valid region of the output page.
  for (p.u=validOutBox.v1.u;  p.u<=validOutBox.v2.u;  ++p.u) {
    for (p.t=validOutBox.v1.t;  p.t<=validOutBox.v2.t;  ++p.t) {
      for (p.c=validOutBox.v1.c;  p.c<=validOutBox.v2.c;  ++p.c) {
        for (p.z=validOutBox.v1.z;  p.z<=validOutBox.v2.z;  ++p.z) {
			for (p.y = validOutBox.v1.y; p.y <= validOutBox.v2.y; ++p.y) {

				p.x = validOutBox.v1.x;
				// Get pointers to row starts of input and output sub-images.
				const T* inVoxel0 = inputSubImage0->getImagePointer(p);
				T*  outVoxel = outputSubImage->getImagePointer(p);


				const MLint rowEnd = validOutBox.v2.x;
				int x;
				int y;
				ImageVector p2 = copy(p);
				// Process all row voxels.
				for (; p.x <= rowEnd; ++p.x, ++outVoxel, ++inVoxel0)
				{
					if (*inVoxel0 > 0){
						if (erosion)
						{

							*outVoxel = *inVoxel0;
							for (x = 0 < p.x - radius ? p.x - radius : 0; x < p.x + radius && x <= validOutBox.v2.x; x++){
								for (y = 0 < p.y - radius ? p.y - radius : 0; y < p.y + radius && y <= validOutBox.v2.y; y++)
								{
									if (inRange(x, y, p.x, p.y)){
										//mlDebug(x);
										p2.x = x;
										p2.y = y;


										if (inputSubImage0->getImageValue(p2) == 0){
											*outVoxel = 0;
											//mlDebug("Removed x:  " << p2.x << "Y: " << p2.y);
										}
									}
								}
							}
						}
						else
						{
							for (x = 0 < p.x - radius ? p.x - radius : 0; x < p.x + radius && x <= validOutBox.v2.x; x++){
								for (y = 0 < p.y - radius ? p.y - radius : 0; y < p.y + radius && y <= validOutBox.v2.y; y++)
								{
									if (inRange(x, y, p.x, p.y)){
										//mlDebug(x);
										p2.x = x;
										p2.y = y;
										*outputSubImage->getImagePointer(p2) = 255;
										mlDebug("Removed x:  " << p2.x << "Y: " << p2.y);
									}
								}
							}
						}
					}


				}
				//*outVoxel = *inVoxel0;

			}
        }
      }
    }
  }
}

ImageVector copy(ImageVector p)
{
	ImageVector result;
	result.u = p.u;
	result.t = p.t;
	result.c = p.c;
	result.z = p.z;
	result.x = p.x;
	result.y = p.y;
	return result;
		
}


bool inRange(int x1, int y1, int x2, int y2)
{

	return ((x1 - x2)* (x1 - x2) + (y1 - y2) *(y1 - y2)) <= (radius * radius);
}

ML_END_NAMESPACE