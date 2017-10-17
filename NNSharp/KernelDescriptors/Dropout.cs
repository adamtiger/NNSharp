using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Dropout : IKernelDescriptor
    {
        public Dropout(double rate, Data2D noiseShape)
        {
            this.rate = rate;
            this.noiseShape = noiseShape;
        }

        public double Rate { get { return rate; } }
        public Data2D NoiseShape { get { return noiseShape; } }

        private double rate;
        private Data2D noiseShape;
    }
}
