using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class LSTMKernel : IKernel
    {
        public void Execute()
        {
            for (int batch = 0; batch < input.GetDimension().b; ++batch)
            {
                h.ToZeros();
                C.ToZeros();

                for (int step = 0; step < input.GetDimension().w; ++step)
                {
                    CalculateI(batch, step);
                    CalculateF(batch, step);
                    CalculateC0(batch, step);
                    CalculateOu(batch, step);
                    ReCalculateC();
                    ReCalculateH();
                }
            }
        }

        protected Data2D input; // batch: batch, channel: input dim, width: timesteps, height: 1
        protected Data2D output; // batch: batch, channel: units (1 x 1 x units x batch)

        protected Data2D wI, wF, wC, wO; // w_: units x input dim x 1 x 1
        protected Data2D uI, uF, uC, uO; // u_: units x units x 1 x 1
        protected Data2D bI, bF, bC, bO; // b_: 1 x units x 1 x 1

        protected Data2D h, C;
        protected Data2D i, f, C0;

        protected ActivationLambda activation, recurrentActivation;

        #region Helper functions

        private void CalculateI(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wI.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wI.GetDimension().w; ++inelm)
                {
                    result += wI[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uI.GetDimension().w; ++inhelm)
                {
                    result += uI[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bI[0, 0, units, 0];

                i[0, 0, units, 0] = result;
            }

            recurrentActivation(i);
        }

        private void CalculateF(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wF.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wF.GetDimension().w; ++inelm)
                {
                    result += wF[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uF.GetDimension().w; ++inhelm)
                {
                    result += uF[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bF[0, 0, units, 0];

                f[0, 0, units, 0] = result;
            }

            recurrentActivation(f);
        }

        private void CalculateC0(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wC.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wC.GetDimension().w; ++inelm)
                {
                    result += wC[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uC.GetDimension().w; ++inhelm)
                {
                    result += uC[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bC[0, 0, units, 0];

                C0[0, 0, units, 0] = result;
            }

            activation(C0);
        }

        private void CalculateOu(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wO.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wO.GetDimension().w; ++inelm)
                {
                    result += wO[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uO.GetDimension().w; ++inhelm)
                {
                    result += uO[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bO[0, 0, units, 0];

                output[0, 0, units, batch] = result;
            }

            recurrentActivation(output);
        }

        private void ReCalculateC()
        {
            for (int units = 0; units < i.GetDimension().c; ++units)
            {
                C[0, 0, units, 0] = 
                    i[0, 0, units, 0] * C0[0, 0, units, 0] + f[0, 0, units, 0] * C[0, 0, units, 0];
            }
        }

        private void ReCalculateH()
        {
            activation(C);

            for (int units = 0; units < h.GetDimension().c; ++units)
            {
                h[0, 0, units, 0] = output[0, 0, units, 0] * C[0, 0, units, 0];
            }
        }

        #endregion
    }
}
