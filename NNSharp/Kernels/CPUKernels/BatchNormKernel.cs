using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class BatchNormKernel : IKernel
    {
        public void Execute()
        {
            dim = input.GetDimension();

            for (int c = 0; c < dim.c; ++c)
            {
                scale = gamma[c] / Math.Sqrt(variance[c] + epsilon);
                offset = beta[c] - scale * bias[c]; // see article (https://arxiv.org/abs/1502.03167)

                for (int b = 0; b < dim.b; ++b)
                {
                    for (int h = 0; h < dim.h; ++h)
                    {
                        for (int w = 0; w < dim.w; ++w)
                        {
                            output[h, w, c, b] = scale * input[h, w, c, b] + offset;
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected List<double> gamma;
        protected List<double> beta;
        protected List<double> bias;
        protected List<double> variance;
        protected double epsilon;

        private Dimension dim;
        private double scale = 0.0;
        private double offset = 0.0;
    }
}
