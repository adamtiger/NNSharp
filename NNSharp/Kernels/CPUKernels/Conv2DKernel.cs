using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class Conv2DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimK = weights.GetDimension();
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            int stH = 0;
            int stV = 0;

            for (int batch = 0; batch < dimO.b; ++batch)
            {
                for (int filter = 0; filter < dimO.c; ++filter)
                {
                    for (int w = 0; w < dimO.w; ++w)
                    {
                        for (int h = 0; h < dimO.h; ++h)
                        {
                            stH = w * strideHorizontal - paddingHorizontal;
                            stV = h * strideVertical - paddingVertical;
                            output[h, w, filter, batch] = 0.0;

                            for (int idxH = stH; idxH < stH + dimK.w; ++idxH)
                            {
                                 for (int idxV = stV; idxV < stV + dimK.h; ++idxV)
                                 {
                                     for (int idxC = 0; idxC < dimK.c; ++idxC)
                                     {
                                        output[h, w, filter, batch] += input[idxV, idxH, idxC, batch] *
                                                                            weights[idxV - stV, idxH - stH, idxC, filter];
                                     }
                                 }
                            }
                            
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;
        protected Data2D weights;

        protected int paddingVertical;
        protected int paddingHorizontal;
        protected int strideVertical;
        protected int strideHorizontal;
    }
}
