using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class GRUKernel : IKernel
    {
        public void Execute()
        {
            output.ToZeros();
            for (int batch = 0; batch < input.GetDimension().b; ++batch)
            {
                h.ToZeros();

                for (int step = 0; step < input.GetDimension().w; ++step)
                {
                    CalculateZ(batch, step);
                    CalculateR(batch, step);
                    CalculateHH(batch, step);
                    ReCalculateH();
                }

                CopyToOutput(batch);
            }
        }

        protected Data2D input; // batch: batch, channel: input dim, width: timesteps, height: 1
        protected Data2D output; // batch: batch, channel: units (1 x 1 x units x batch)

        protected Data2D wZ, wR, wHH; // w_: units x input dim x 1 x 1
        protected Data2D uZ, uR, uHH; // u_: units x units x 1 x 1
        protected Data2D bZ, bR, bHH; // b_: 1 x units x 1 x 1

        protected Data2D h;
        protected Data2D z, r, hh;

        protected ActivationLambda activation, recurrentActivation;


        # region Helper functions 

        private void CalculateZ(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wZ.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wZ.GetDimension().w; ++inelm)
                {
                    result += wZ[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uZ.GetDimension().w; ++inhelm)
                {
                    result += uZ[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bZ[0, 0, units, 0];

                z[0, 0, units, 0] = result;
            }

            recurrentActivation(z);
        }

        private void CalculateR(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wR.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wR.GetDimension().w; ++inelm)
                {
                    result += wR[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uR.GetDimension().w; ++inhelm)
                {
                    result += uR[units, inhelm, 0, 0] * h[0, 0, inhelm, 0];
                }

                result += bR[0, 0, units, 0];

                r[0, 0, units, 0] = result;
            }

            recurrentActivation(r);
        }

        private void CalculateHH(int batch, int step)
        {
            double result = 0;
            for (int units = 0; units < wHH.GetDimension().h; ++units)
            {
                result = 0;
                for (int inelm = 0; inelm < wHH.GetDimension().w; ++inelm)
                {
                    result += wHH[units, inelm, 0, 0] * input[0, step, inelm, batch];
                }

                for (int inhelm = 0; inhelm < uHH.GetDimension().w; ++inhelm)
                {
                    result += uHH[units, inhelm, 0, 0] * h[0, 0, inhelm, 0] * r[0, 0, inhelm, 0];
                }

                result += bHH[0, 0, units, 0];

                hh[0, 0, units, 0] = result;
            }

            activation(hh);
        }

        private void ReCalculateH()
        {
            for (int units = 0; units < h.GetDimension().c; ++units)
            {
                h[0, 0, units, 0] = z[0, 0, units, 0] * h[0, 0, units, 0] + 
                    (1-z[0, 0, units, 0]) * hh[0, 0, units, 0];
            }
        }

        private void CopyToOutput(int batch)
        {
            for (int units = 0; units < output.GetDimension().c; ++units)
            {
                output[0, 0, units, batch] = h[0, 0, units, 0];
            }
        }

        #endregion
    }
}
