using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class LeakyReLuLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is LeakyReLu) {
                LeakyReLu leakyrelu = descriptor as LeakyReLu;

                return new LeakyReLuLayer(leakyrelu.Alpha);
            }

            return null;
        }
    }
}
