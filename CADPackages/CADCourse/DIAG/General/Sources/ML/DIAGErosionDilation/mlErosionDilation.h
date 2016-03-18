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


#ifndef __mlErosionDilation_H
#define __mlErosionDilation_H


// Local includes
#include "DIAGErosionDilationSystem.h"

// ML includes
#include <mlModuleIncludes.h>

ML_START_NAMESPACE


//! 
class DIAGEROSIONDILATION_EXPORT ErosionDilation : public Module
{
public:

  //! Constructor.
  ErosionDilation();

  //! Handles field changes of the field \p field.
  virtual void handleNotification (Field* field);

  // ----------------------------------------------------------
  //! \name Image processing methods.
  //@{
  // ----------------------------------------------------------

  //! Sets properties of the output image at output \p outputIndex.
  virtual void calculateOutputImageProperties(int outputIndex, PagedImage* outputImage);

  //! Returns the input image region required to calculate a region of an output image.
  //! \param inputIndex        The input of which the regions shall be calculated.
  //! \param outputSubImageBox The region of the output image for which the required input region
  //!                          shall be calculated.
  //! \param outputIndex       The index of the output image for which the required input region
  //!                          shall be calculated.
  //! \return Region of input image needed to compute the region \p outputSubImageBox on output \p outputIndex.
  virtual SubImageBox calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex);

  //! Calculates page \p outputSubImage of output image with index \p outputIndex by using \p inputSubImages.
  //! \param outputSubImage The sub-image of output image \p outputIndex calculated from \p inputSubImges.
  //! \param outputIndex    The index of the output the sub-image is calculated for.
  //! \param inputSubImages Array of sub-image(s) of the input(s) whose extents were specified
  //!                       by calculateInputSubImageBox. Array size is given by getNumInputImages().
  virtual void calculateOutputSubImage(SubImage* outputSubImage, int outputIndex, SubImage* inputSubImages);

  //! Method template for type-specific page calculation. Called by calculateOutputSubImage().
  //! \param outputSubImage The typed sub-image of output image \p outputIndex calculated from \p inputSubImages.
  //! \param outputIndex    The index of the output the sub-image is calculated for.
  //! \param inSubImg0 Temporary typed sub-image of input 0.
  template <typename T>
  void calculateOutputSubImage (TSubImage<T>* outputSubImage, int outputIndex
                               , TSubImage<T>* inputSubImage0
                               );
  //@}

private:

  // ----------------------------------------------------------
  //! \name Module field declarations
  //@{
  // ----------------------------------------------------------

  //! 
  IntField* _radiusFld;
  //! Erosion instead ofdilation when checked
  BoolField* _erosionFld;
  //@}

  // Implements interface for the runtime type system of the ML.
  ML_MODULE_CLASS_HEADER(ErosionDilation)
};


ML_END_NAMESPACE

#endif // __mlErosionDilation_H