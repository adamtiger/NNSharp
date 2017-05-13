using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class MaxPool1DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            int stH = 0;

            for (int batch = 0; batch < dimO.b; ++batch)
            {
                for (int channel = 0; channel < dimO.c; ++channel)
                {
                    for (int l = 0; l < dimO.w; ++l)
                    {
                        stH = l * stride - padding;
                        output[0, l, channel, batch] = Double.MinValue;

                        for (int idx = stH; idx < stH + kernelDim.w; ++idx)
                        {
                            output[0, l, channel, batch] = Math.Max(input[0, idx, channel, batch],
                                                                output[0, l, channel, batch]);
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected int padding;
        protected int stride;

        protected Dimension kernelDim;
    }
}
