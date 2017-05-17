using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class GlobalAvgPool1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is GlobalAvgPooling1D)
            {
                GlobalAvgPooling1D pool = descriptor as GlobalAvgPooling1D;

                ILayer layer = new GlobalAvgPool1DLayer();

                return layer;
            }

            return null;
        }
    }
}
