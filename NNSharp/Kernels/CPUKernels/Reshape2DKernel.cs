using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class Reshape2DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();

            int B1 = dimI.h * dimI.w * dimI.c;
            int B2 = dimO.h * dimO.w * dimO.c;

            int H1 = dimI.w * dimI.c;
            int H2 = dimO.w * dimO.c;

            int W1 = dimI.c;
            int W2 = dimO.c;

            int virtualIdx = 0;
            int cc, bb, hh, ww;

            for (int b = 0; b < dimI.b; ++b)
            {
                for (int h = 0; h < dimI.h; ++h)
                {
                    for (int w = 0; w < dimI.w; ++w)
                    {
                        for (int c = 0; c < dimI.c; ++c)
                        {
                            virtualIdx = b * B1 + h * H1 + w * W1 + c;

                            bb = (int)Math.Floor((double)virtualIdx / B2);
                            virtualIdx = virtualIdx % B2;

                            hh = (int)Math.Floor((double)virtualIdx / H2);
                            virtualIdx = virtualIdx % H2;

                            ww = (int)Math.Floor((double)virtualIdx / W2);
                            cc = virtualIdx % W2;

                            output[hh, ww, cc, bb] = input[h, w, c, b];
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output; 
    }
}
