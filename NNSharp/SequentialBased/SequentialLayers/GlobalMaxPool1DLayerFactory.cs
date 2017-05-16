using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class GlobalMaxPool1DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is GlobalMaxPooling1D)
            {
                GlobalMaxPooling1D pool = descriptor as GlobalMaxPooling1D;

                ILayer layer = new GlobalMaxPool1DLayer();

                return layer;
            }

            return null;
        }
    }
}
