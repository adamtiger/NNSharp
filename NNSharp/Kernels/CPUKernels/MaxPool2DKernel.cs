using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class MaxPool2DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            int stH = 0;
            int stV = 0;

            for (int batch = 0; batch < dimO.b; ++batch)
            {
                for (int channel = 0; channel < dimO.c; ++channel)
                {
                    for (int w = 0; w < dimO.w; ++w)
                    {
                        for (int h = 0; h < dimO.h; ++h)
                        {
                            stH = w * strideHorizontal - paddingHorizontal;
                            stV = h * strideVertical - paddingVertical;
                            output[h, w, channel, batch] = 0.0;

                            for (int idxH = stH; idxH < stH + kernelDim.w; ++idxH)
                            {
                                for (int idxV = stV; idxV < stV + kernelDim.h; ++idxV)
                                {
                                     output[h, w, channel, batch] = Math.Max(input[idxV, idxH, channel, batch],
                                                                         output[h, w, channel, batch]);
                                }
                            }

                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected int paddingVertical;
        protected int paddingHorizontal;
        protected int strideVertical;
        protected int strideHorizontal;

        protected Dimension kernelDim;

    }
}
