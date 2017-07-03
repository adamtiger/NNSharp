using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels
{
    public delegate void ActivationLambda(IData data);

    public interface IKernel
    {
        void Execute();
    }
}
