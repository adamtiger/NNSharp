using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class SoftmaxKernel : IKernel
    {
        public void Execute()
        {
            SoftmaxLambda(data);
        }

        public static void SoftmaxLambda(IData data)
        {
            Data2D dat = data as Data2D;
            for (int b = 0; b < dat.GetDimension().b; ++b)
            {
                double sum = 0.0;
                for (int i = 0; i < dat.GetDimension().c; ++i)
                {
                    sum += Math.Exp(dat[0, 0, i, b]);
                }

                for (int i = 0; i < dat.GetDimension().c; ++i)
                {
                    dat[0, 0, i, b] = Math.Exp(dat[0, 0, i, b]) / sum;
                }
            }
        }

        protected Data2D data;
    }
}
