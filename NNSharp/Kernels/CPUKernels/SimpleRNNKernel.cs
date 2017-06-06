using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class SimpleRNNKernel : IKernel
    {
        public void Execute()
        {
            for (int b = 0; b < input.GetDimension().b; ++b)
            {
                prevOutput.ToZeros();

                for (int step = 0; step < input.GetDimension().w; ++step)
                {
                    InputDotKernel(input, step, b, kernel, h);
                    HAddBias(h, bias, h);
                    HAddPrevoutDotRecKernel(h, prevOutput, recurrentKernel, output, b);
                    Copy(prevOutput, output, b);
                    Activation(prevOutput);
                }
            }

            Activation(output);
        }

        protected Data2D input; // batch: batch, channel: input dim, width: timesteps, height: 1
        protected Data2D output; // batch: batch, channel: units

        protected Data2D kernel; // batch: 1, channel: units, width: input dim, height: 1
        protected Data2D recurrentKernel; // batch: 1, channel: units, width: units, height: 1
        protected Data2D bias; // batch: 1, channel: units, width: 1, height: 1

        protected ActivationLambda lambda;

        protected Data2D h;
        protected Data2D prevOutput;


        #region Matrix multiplications

        private void InputDotKernel(Data2D input, int step, int batch, Data2D kernel, Data2D result)
        {
            for (int u = 0; u < kernel.GetDimension().c; ++u)
            {
                result[0, 0, u, 0] = 0;
                for (int d = 0; d < input.GetDimension().c; ++d)
                {
                    result[0, 0, u, 0] += input[0, step, d, batch] * kernel[0, d, u, 0];
                }
            }
        }

        private void HAddBias(Data2D h, Data2D bias, Data2D result)
        {
            for (int d = 0; d < h.GetDimension().c; ++d)
            {
                result[0, 0, d, 0] = h[0, 0, d, 0] + bias[0, 0, d, 0];
            }
        }

        private void HAddPrevoutDotRecKernel(Data2D h, Data2D prevOut, Data2D recKern, Data2D result, int batch)
        {
            for (int u = 0; u < kernel.GetDimension().c; ++u)
            {
                result[0, 0, u, batch] = 0;
                for (int d = 0; d < prevOut.GetDimension().c; ++d)
                {
                    result[0, 0, u, batch] += prevOut[0, 0, d, 0] * recKern[0, d, u, 0];
                }

                result[0, 0, u, batch] = h[0, 0, u, 0] + result[0, 0, u, batch];
            }
        }

        private void Copy(Data2D prevOut, Data2D output, int batch)
        {
            for (int u = 0; u < prevOutput.GetDimension().c; ++u)
            {
                prevOutput[0, 0, u, 0] = output[0, 0, u, batch];
            }
        }

        private void Activation(Data2D data)
        {
            lambda(data); 
        }

        #endregion

    }
}
